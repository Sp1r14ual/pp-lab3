#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <time.h>
#include <omp.h>

void set_texture();

typedef struct { unsigned char r, g, b; } rgb_t;
rgb_t **tex = 0;
int gwin;
GLuint texture;
int width, height;
int tex_w, tex_h;
double scale = 1./256;
double cx = -.6, cy = 0;
int color_rotate = 0;
int saturation = 1;
int invert = 0;
int max_iter = 256;

// Объявим отдельный массив для хранения итераций
unsigned short **iters;

void render() {
    double x = (double)width / tex_w,
           y = (double)height / tex_h;

    glClear(GL_COLOR_BUFFER_BIT);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glBindTexture(GL_TEXTURE_2D, texture);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2i(0, 0);
    glTexCoord2f(x, 0); glVertex2i(width, 0);
    glTexCoord2f(x, y); glVertex2i(width, height);
    glTexCoord2f(0, y); glVertex2i(0, height);
    glEnd();

    glFlush();
    glFinish();
}

int dump = 1;
void screen_dump() {
    char fn[100];
    sprintf(fn, "screen%03d.ppm", dump++);
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    for (int i = height - 1; i >= 0; i--)
        fwrite(tex[i], 3, width, fp);
    fclose(fp);
    printf("%s written\n", fn);
}

void keypress(unsigned char key, int x, int y) {
    switch (key) {
        case 'q':
            glutDestroyWindow(gwin);
            exit(0);
        case 27:
            scale = 1./256; cx = -.6; cy = 0;
            break;
        case 'r':
            color_rotate = (color_rotate + 1) % 6;
            break;
        case '>': case '.':
            max_iter += 128;
            if (max_iter > 1 << 15) max_iter = 1 << 15;
            printf("max iter: %d\n", max_iter);
            break;
        case '<': case ',':
            max_iter -= 128;
            if (max_iter < 128) max_iter = 128;
            printf("max iter: %d\n", max_iter);
            break;
        case 'c':
            saturation = 1 - saturation;
            break;
        case 's':
            screen_dump();
            return;
        case 'z':
            max_iter = 4096;
            break;
        case 'x':
            max_iter = 128;
            break;
        case ' ':
            invert = !invert;
            break;
    }
    set_texture();
}

void hsv_to_rgb(int hue, int min, int max, rgb_t *p) {
    if (min == max) max = min + 1;
    if (invert) hue = max - (hue - min);
    if (!saturation) {
        p->r = p->g = p->b = 255 * (max - hue) / (max - min);
        return;
    }
    double h = fmod(color_rotate + 4.0 * (hue - min) / (max - min), 6);
    double c = 255 * saturation;
    double X = c * (1 - fabs(fmod(h, 2) - 1));

    p->r = p->g = p->b = 0;

    switch ((int)h) {
        case 0: p->r = c; p->g = X; break;
        case 1: p->r = X; p->g = c; break;
        case 2: p->g = c; p->b = X; break;
        case 3: p->g = X; p->b = c; break;
        case 4: p->r = X; p->b = c; break;
        default: p->r = c; p->b = X;
    }
}

void calc_mandel() {
    int min = max_iter, max = 0;

    // Вычисляем итерации
    // #pragma omp parallel for
    #pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < height; i++) {
        double y = (i - height/2) * scale + cy;
        for (int j = 0; j < width; j++) {
            double x = (j - width/2) * scale + cx;
            double zx = 0.0, zy = 0.0, zx2 = 0.0, zy2 = 0.0;
            int iter = 0;

            // Проверка на принадлежность к главной кардиоиде или кругу
            double q = (x - 0.25) * (x - 0.25) + y * y;
            if (q * (q + (x - 0.25)) < 0.25 * y * y || (x + 1)*(x + 1) + y*y < 0.0625) {
                iter = max_iter;
            } else {
                while (iter < max_iter && zx2 + zy2 < 4.0) {
                    zy = 2 * zx * zy + y;
                    zx = zx2 - zy2 + x;
                    zx2 = zx * zx;
                    zy2 = zy * zy;
                    iter++;
                }
            }
            iters[i][j] = iter;
            if (iter < min) min = iter;
            if (iter > max) max = iter;
        }
    }

    // Преобразование итераций в цвета
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            hsv_to_rgb(iters[i][j], min, max, &tex[i][j]);
        }
    }
}

void alloc_tex() {
    // Находим ближайшие степени двойки
    tex_w = 1;
    while (tex_w < width) tex_w <<= 1;
    tex_h = 1;
    while (tex_h < height) tex_h <<= 1;

    // Выделяем память под текстуру и массив итераций
    tex = realloc(tex, tex_h * sizeof(rgb_t*) + tex_h * tex_w * sizeof(rgb_t));
    iters = realloc(iters, tex_h * sizeof(unsigned short*) + tex_h * tex_w * sizeof(unsigned short));

    rgb_t *tex_data = (rgb_t*)(tex + tex_h);
    unsigned short *iters_data = (unsigned short*)(iters + tex_h);
    for (int i = 0; i < tex_h; i++) {
        tex[i] = tex_data + i * tex_w;
        iters[i] = iters_data + i * tex_w;
    }
}

void set_texture() {
    alloc_tex();

    clock_t start = clock();
    calc_mandel();
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;

    printf("Algorithm Time: %f\n\n", seconds);


    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_w, tex_h, 0, GL_RGB, GL_UNSIGNED_BYTE, tex[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    render();
}

void mouseclick(int button, int state, int x, int y) {
    if (state != GLUT_UP) return;
    cx += (x - width/2) * scale;
    cy -= (y - height/2) * scale;
    if (button == GLUT_LEFT_BUTTON) scale /= 2;
    else if (button == GLUT_RIGHT_BUTTON) scale *= 2;
    set_texture();
}

void resize(int w, int h) {
    width = w;
    height = h;
    glViewport(0, 0, w, h);
    glOrtho(0, w, 0, h, -1, 1);
    set_texture();
}

void init_gfx(int *c, char **v) {
    glutInit(c, v);
    glutInitDisplayMode(GLUT_RGB);
    glutInitWindowSize(1280, 720);
    gwin = glutCreateWindow("Mandelbrot");
    glutDisplayFunc(render);
    glutKeyboardFunc(keypress);
    glutMouseFunc(mouseclick);
    glutReshapeFunc(resize);
    glGenTextures(1, &texture);
    set_texture();
}

int main(int c, char **v) {
    // clock_t start = clock();
    init_gfx(&c, v);
    // clock_t end = clock();
    // float seconds = (float)(end - start) / CLOCKS_PER_SEC;

    // printf("Algorithm Time: %f\n\n", seconds);

    printf("Controls:\n"
           "r: Rotate colors\n"
           "c: Toggle color saturation\n"
           "s: Save screenshot\n"
           "<, >: Adjust iterations\n"
           "Mouse: Zoom\n");
    glutMainLoop();
    return 0;
}