#ifndef GLES2_HELPER
#define GLES2_HELPER
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <fcntl.h>
#include <errno.h>
#include <EGL/egl.h>
#include <gbm.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

/*
* This structure is to control EGL and GBM
*/

struct egl_gbm_controller{
    EGLDisplay display;
	EGLConfig *configs;
	EGLContext context;
	EGLSurface surface;
    EGLint majorVersion;
    EGLint minorVersion;
    EGLint numConfigs;
    EGLint count;
    struct gbm_device *gbm_dev;
	struct gbm_surface *gbm_surface;
    int fd;

};
typedef struct egl_gbm_controller egl_gbm_controller;
/*
* @brief This data structure is to hold gpu data
* @param text_id: texture id
* @param textSize: texture size
* @param col_order: ensure 1d array transfrom into 2d matrix in collumn order
* @param fdata: hold the data for the gpu to process
*/
struct gles2_data{
    int text_id;
    size_t textSize, n;
    bool col_order;
    float * fdata;
    
};
typedef struct gles2_data gles2_data;
/*
* @brief This structure is to manage computing for OpenGL ES2
* @param ver_shader: vertex shader
* @param frag_shader: fragment shader
* @param pro_id: OpenGL ES program id
* @param num_text: number of texture to process in fragment shader
* @param ver_id, frag_id: id of vertex shader and fragment shader
*/
struct gles2_controller{
    const char*  ver_shader; // vertex shader
    const char*  frag_shader; // fragment shader
    int pro_id;
    int fb_id;
    int num_text;
    GLint ver_id, frag_id;
   
};
typedef struct gles2_controller gles2_controller;
static const EGLint configAttribs[] = {
    EGL_BUFFER_SIZE, 32,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
    EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
    EGL_NONE
    };
static const EGLint contextAttribs[] = {
    EGL_CONTEXT_CLIENT_VERSION, 2,
    EGL_NONE};


/*
* @brief function to check error when initialize egl, 
* this function I modify from rpi_without_x code: https://github.com/matusnovak/rpi-opengl-without-x
*/
#ifdef __cplusplus
extern "C" {
#endif
extern const char* check_egl_error();
/*
* @brief function to initialize EGL, GBM to prepare for OpenGL ES2
* @param file: the path to the GPU file: here I use "/dev/dri/renderD128" in my test file
*/
extern void gles2_init(egl_gbm_controller* my_controller, const char * file);
/*
* @brief function to clean EGL, GBM 
*/
extern void gles2_destroy(egl_gbm_controller* my_controller);
/*
* @brief function to check for matching config between EGL and GBM. It used by function gles2_make_surface
*/
extern int gles2_matchconfig_egl(egl_gbm_controller* my_controller, uint32_t gbm_format);
/*
* @brief function to prepare surface to contain gpu data
* @details This function is crucial for computing. Anything change on parameters may result in lost data
* @param w: width of draw surface
* @param h: height of draw surface
*/
extern void gles2_make_surface(egl_gbm_controller* my_controller, uint32_t w, uint32_t h);
/*
* @brief Function to destroy surface
*/
extern void gles2_destroy_surface(egl_gbm_controller* my_controller);
/*
* @brief function to prepare data for gpu to process
* @param data: array of floating point data
* @param n: size of array
*/
extern gles2_data* gles2_make_farr(float *data, size_t n);
/*
* @brief create framebuffer to retrieve the result after computing
* @param gpu_data: the data to get the result after computing
*/
extern void gles2_make_fbo(gles2_controller* my_controller, gles2_data* gpu_data);
/*
* @brief function to push data to gpu memory
*/
extern void gles2_push_farr( gles2_controller* my_controller, gles2_data* gpu_data, const char*name , bool is_input);
/*
* @brief function to push single float number to gpu memory
* @param gpu_data: floating point number
* @param name: name of floating point number 
*/
extern void gles2_push_float(gles2_controller* my_controller, float gpu_data, const char*name );
extern void gles2_push_int(gles2_controller* my_controller, int gpu_data, const char*name );
/*
* @brief function to retrieve data from gpu memory back
* @param data: data to contain result of gpu data, it can be NULL
* @param n: number of floating point number to retrieve for data
* @param gpu_data: this is to get data back from gpu, it must not NULL  
*/
extern void gles2_pull_farr( float* data, size_t n,gles2_data* gpu_data);
/*
* @brief function to destroy frame buffer holding result after using GPU for computing
*/
extern void gles2_destroy_fbo(gles2_controller* my_controller);
/*
* @brief function to free data in gpu memory
*/
extern void gles2_free_dev_farr(gles2_data* gpu_data);
/*
* @brief function to free data in host memory
*/
extern void gles2_free_host_farr(float *data);
/*
* @brief function to check for compile error of shader. It is used by function gles2_build
*/
extern void gles2_check_compile_error(unsigned int shader_id, const char* type);
/*
* @brief function to build program for computing
*/
extern void gles2_build(gles2_controller* my_controller);
/*
* @brief function to conduct gpu computation
*/
extern void gles2_compute(gles2_controller* my_controller);
/*
* @brief function to map texture coordinate. It is crucial for GPU computing.
*/
extern void gles2_setViewport(uint32_t width, uint32_t height);
/*
* @brief function to check for OpenGL ES2 error 
*/
extern int gles2_checkerror();
#ifdef __cplusplus
}
#endif
#endif