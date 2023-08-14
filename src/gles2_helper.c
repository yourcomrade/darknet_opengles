#include "gles2_helper.h"
const char* check_egl_error(){
    switch (eglGetError())
    {
    case EGL_SUCCESS:
        return "Success.";
    case EGL_NOT_INITIALIZED:
        return "EGL is not initialized, or could not be initialized, for the "
               "specified EGL display connection.";
    case EGL_BAD_ACCESS:
        return "EGL cannot access a requested resource. ";
    case EGL_BAD_ALLOC:
        return "EGL failed to allocate resources for the requested operation.";
    case EGL_BAD_ATTRIBUTE:
        return "An unrecognized attribute or attribute value was passed in the "
               "attribute list.";
    case EGL_BAD_CONTEXT:
        return "An EGLContext argument does not name a valid EGL rendering "
               "context.";
    case EGL_BAD_CONFIG:
        return "An EGLConfig argument does not name a valid EGL frame buffer "
               "configuration.";
    case EGL_BAD_CURRENT_SURFACE:
        return "The current surface of the calling thread is a window, pixel "
               "buffer or pixmap that is no longer valid.";
    case EGL_BAD_DISPLAY:
        return "An EGLDisplay argument does not name a valid EGL display "
               "connection.";
    case EGL_BAD_SURFACE:
        return "An EGLSurface argument does not name a valid surface (window, "
               "pixel buffer or pixmap) configured for GL rendering.";
    case EGL_BAD_MATCH:
        return "Arguments are inconsistent (for example, a valid context "
               "requires buffers not supplied by a valid surface).";
    case EGL_BAD_PARAMETER:
        return "One or more argument values are invalid.";
    case EGL_BAD_NATIVE_PIXMAP:
        return "A NativePixmapType argument does not refer to a valid native "
               "pixmap.";
    case EGL_BAD_NATIVE_WINDOW:
        return "A NativeWindowType argument does not refer to a valid native "
               "window.";
    case EGL_CONTEXT_LOST:
        return "A power management event has occurred. The application must "
               "destroy all contexts and reinitialise OpenGL ES state and "
               "objects to continue rendering.";
    default:
        break;
    }
    return "Unknown error!";
}


void gles2_init(egl_gbm_controller* my_controller, const char * file){
    //initialize 
    my_controller->fd = open(file,O_RDWR);
    if(my_controller->fd < 0){
        perror("Cannot open device driver file of graphic");
        abort();
    }
    my_controller->gbm_dev = gbm_create_device(my_controller->fd);
    if(my_controller->gbm_dev == NULL){
        perror("Cannot create gbm device");
        abort();
    }
    //check for display
    my_controller->display = eglGetDisplay((EGLNativeDisplayType)(my_controller->gbm_dev));
    if(my_controller->display == EGL_NO_DISPLAY){
        perror("eglGetDisplay fails!");
        gbm_device_destroy(my_controller->gbm_dev);
        close(my_controller->fd);
        abort();
    }
    //initialize program
    if(eglInitialize(my_controller->display,&(my_controller->majorVersion),
    &(my_controller->minorVersion)) == EGL_FALSE){
       fprintf(stderr, "Failed to get EGL version! Error: %s\n",
                check_egl_error());
        gles2_destroy(my_controller);
        abort();
    }
    printf("EGL init with version %d.%d\n", my_controller->majorVersion, my_controller->minorVersion);
    //Bind API
    if(eglBindAPI(EGL_OPENGL_ES_API) == EGL_FALSE){
        fprintf(stderr,"eglBindApi failed! Error: %s\n",check_egl_error());
        gles2_destroy(my_controller);
        abort();
    }
    //Initialize config

    if(eglGetConfigs(my_controller->display, NULL, 0, &(my_controller->count)) == EGL_FALSE){
        fprintf(stderr, "Get egl config failed! Error: %s\n", check_egl_error());
        gles2_destroy(my_controller);
        abort();
    }

    my_controller->configs = calloc(my_controller->count, sizeof(EGLConfig));
    if(my_controller->configs == NULL){
        perror("Cannot allocate memory for egl config!");
        gles2_destroy(my_controller);
        abort();
    }

    if(eglChooseConfig(my_controller->display, configAttribs, (my_controller->configs),
    my_controller->count , &(my_controller->numConfigs)) == EGL_FALSE){
        fprintf(stderr, "eglChooseConfig failed! Error: %s\n", check_egl_error());
        gles2_destroy(my_controller);
        abort();
    }
    
   
}

void gles2_destroy(egl_gbm_controller* my_controller){
    free(my_controller->configs);
    eglTerminate (my_controller->display);
    gbm_device_destroy (my_controller->gbm_dev);
    close (my_controller->fd);
}

int gles2_matchconfig_egl(egl_gbm_controller* my_controller, uint32_t gbm_format){
    EGLint id_format;
    for(int i = 0; i < my_controller->numConfigs; i++){
        if(eglGetConfigAttrib(my_controller->display,
        my_controller->configs[i], EGL_NATIVE_VISUAL_ID, &id_format ) == EGL_FALSE){
            printf("Could not get configs\n");
            continue;
        }
        if(id_format == gbm_format){
            return i;
        }
    }
    return -1;
}

void gles2_make_surface( egl_gbm_controller* my_controller, uint32_t width, uint32_t height){
  
    my_controller->gbm_surface = gbm_surface_create(my_controller->gbm_dev, width, height, 
    GBM_FORMAT_ARGB8888, GBM_BO_USE_RENDERING);
    if(my_controller->gbm_surface == NULL){
        perror("Cannot create gbm surface!");
        gles2_destroy(my_controller);
        abort();
    }
    printf("Done make gbm surface");
    int config_id = gles2_matchconfig_egl(my_controller, GBM_FORMAT_ARGB8888);
    if(config_id < 0){
        perror("Fail to find matching config!");
        gbm_surface_destroy(my_controller->gbm_surface);
        gles2_destroy(my_controller);
        abort();
    }
     //Create context
    my_controller->context = eglCreateContext(my_controller->display, my_controller->configs[config_id], 
    EGL_NO_CONTEXT, contextAttribs);
    if(my_controller->context == EGL_NO_CONTEXT){
        fprintf(stderr, "eglCreateContext failed! Error: %s\n", check_egl_error());
        gbm_surface_destroy(my_controller->gbm_surface);
        gles2_destroy(my_controller);
        abort();
    }
    printf("Done make egl context\n");

    my_controller->surface = eglCreateWindowSurface(my_controller->display, my_controller->configs[config_id],
    (EGLNativeWindowType)my_controller->gbm_surface, NULL);
    if(my_controller->surface == EGL_NO_SURFACE){
        fprintf(stderr, "Cannot create window surface! Error: %s\n",check_egl_error());
        gbm_surface_destroy(my_controller->gbm_surface);
        gles2_destroy(my_controller);
        abort();
    }
    printf("Done make surface\n");
    
    if(eglMakeCurrent(my_controller->display, my_controller->surface,
    my_controller->surface, my_controller->context) == EGL_FALSE){
        fprintf(stderr, "Cannot bind current thread to compute! Error: %s\n", check_egl_error());
       
        gles2_destroy_surface(my_controller);
        abort();
    }
     
}

void gles2_destroy_surface(egl_gbm_controller* my_controller){
    if(eglDestroySurface(my_controller->display, my_controller->surface) == EGL_FALSE){
        fprintf(stderr, "Cannot destroy surface! Error: %s\n", check_egl_error());
        gles2_destroy(my_controller);
        abort();
    }
    gbm_surface_destroy(my_controller->gbm_surface);
    if(eglDestroyContext (my_controller->display, my_controller->context) == EGL_FALSE){
        fprintf(stderr, "Cannot destroy context! Error: %s\n", check_egl_error());
        gles2_destroy(my_controller);
        abort();
    }
    if(eglMakeCurrent(my_controller->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT) == EGL_FALSE ){
        fprintf(stderr, "Cannot release current context! Error: %s\n", check_egl_error());
        gles2_destroy(my_controller);
        abort();
    }
    
}

gles2_data * gles2_make_farr(float * data, size_t n){
    float *gpu_data = calloc(n, sizeof(float));
    if(gpu_data == NULL){
        perror("Cannot allocate memory for gpu data\n");
        abort();
    }
    gles2_data * my_data = calloc(1, sizeof(gles2_data));
    if(my_data == NULL){
        perror("Cannot create data structure for gles2_data");
        free(gpu_data);
        abort();
    }
    my_data->fdata = gpu_data;
    my_data->n = n;
    //Calculate texture dimension
    my_data->textSize = (size_t)floor(sqrt(n)) + 1;
    GLint maxsize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxsize);
    printf("Max texture size support: %d\n", maxsize);
    if(my_data->textSize > maxsize){
        perror("the data is too large to fit to texture size");
        free(gpu_data);
        free(my_data);
        abort();
    }

    if(data!=NULL){
        for(size_t i = 0; i < n; i++){
            my_data->fdata[i] = data[i];
        }
    }
   
    return my_data;

}
int gles2_checkerror(){
    GLenum errorcode = glGetError();
    const char* error= NULL;
    if(errorcode != GL_NO_ERROR){
        switch(errorcode){
            case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
            case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
            case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
            case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        fprintf(stderr,"OpenGL ES error: %s\n",error);
        return -1;
    }
    return 1;
}
void gles2_push_farr( gles2_controller* my_controller, gles2_data* gpu_data, const char*name, bool is_input){
    printf("push data\n");
    glGenTextures(1,&(gpu_data->text_id));
    if(is_input){
        
        glActiveTexture(GL_TEXTURE0 + my_controller->num_text);
        
    }
    
  
    glBindTexture(GL_TEXTURE_2D,gpu_data->text_id);
  
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if(!is_input){
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, gpu_data->textSize, gpu_data->textSize, 
    0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        gles2_checkerror();
        return;
    }
   
    union convert{
        unsigned char u[4];
        float f;
    };
    union convert my_convert;
    unsigned char *temp_data = calloc((gpu_data->textSize)*(gpu_data->textSize)*4, sizeof(unsigned char));
    if(temp_data == NULL){
        perror("Cannot allocate memory for temporary data to convert float to unsigned char\n");
        abort();
    }
    size_t j = 0;
   
    for(size_t i = 0; i < gpu_data->n; i++){
        my_convert.f = gpu_data->fdata[i];
        temp_data[j] = my_convert.u[0];
        temp_data[j + 1] = my_convert.u[1];
        temp_data[j + 2] = my_convert.u[2];
        temp_data[j + 3] = my_convert.u[3];
        
        j += 4;
    }
 
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, gpu_data->textSize, gpu_data->textSize, 
    0, GL_RGBA, GL_UNSIGNED_BYTE, temp_data);
    gles2_checkerror();
    glUniform1i(glGetUniformLocation(my_controller->pro_id, name), my_controller->num_text);
  
    gles2_checkerror();
    glFinish();
    my_controller->num_text++;
    free(temp_data);
}
void gles2_push_float(gles2_controller* my_controller, float gpu_data, const char*name ){
    glUniform1f(glGetUniformLocation(my_controller->pro_id, name), gpu_data);
    gles2_checkerror();
}
void gles2_push_int(gles2_controller* my_controller, int gpu_data, const char*name ){
    glUniform1i(glGetUniformLocation(my_controller->pro_id, name), gpu_data);
    gles2_checkerror();
}
void gles2_pull_farr(float* data, size_t n, gles2_data* gpu_data){
    printf("pull data\n");
    if(gpu_data->n < n){
        perror("The size n to copy is larger than gpu data");
        return;
    }
    unsigned char* temp_arr = (unsigned char*)calloc((gpu_data->textSize)*(gpu_data->textSize)*4, sizeof(unsigned char));
    if(temp_arr == NULL){
        perror("Cannot allocate memory for temporary array\n");
        return;
    }

    glReadPixels(0, 0, gpu_data->textSize, gpu_data->textSize, GL_RGBA, GL_UNSIGNED_BYTE, temp_arr);
    gles2_checkerror();
  
    union convert{
        unsigned char u[4];
        float f;
    };
    union convert my_convert;
    size_t j = 0;
    for(size_t i = 0; i < (gpu_data->n)*4; i += 4){
        my_convert.u[0] = temp_arr[i];
        my_convert.u[1] = temp_arr[i + 1];
        my_convert.u[2] = temp_arr[i + 2];
        my_convert.u[3] = temp_arr[i + 3];
        gpu_data->fdata[j] = my_convert.f;
       
        j++;
    }
    free(temp_arr);
    if(data != NULL)
    {
        for(size_t i = 0; i < n; i++){
            data[i] = gpu_data->fdata[i];
        }
    }

}
void gles2_free_dev_farr(gles2_data* gpu_data){
    free(gpu_data->fdata);
    gpu_data->n = 0;
    gpu_data->textSize = 0;
    glDeleteTextures(1, &(gpu_data->text_id));
    free(gpu_data);
    
}

void gles2_make_fbo(gles2_controller* my_controller, gles2_data* gpu_data){
    
    glGenFramebuffers(1,&(my_controller->fb_id));
    glBindFramebuffer(GL_FRAMEBUFFER,my_controller->fb_id);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,gpu_data->text_id,0);
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
   

}
void gles2_destroy_fbo(gles2_controller* my_controller){
    glDeleteFramebuffers (1,&(my_controller->fb_id));
   
}

void gles2_free_host_farr(float *data){
    free(data);
}
void gles2_check_compile_error(unsigned int shader_id, const char* type){
    int success;
    char infoLog[1024];
    if (strcmp(type,"PROGRAM") != 0)
    {
        glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(shader_id, 1024, NULL, infoLog);
            fprintf(stderr, "ERROR::SHADER_COMPILATION_ERROR of type: %s\n%s\n", type, infoLog);
        }
    }
    else
    {
        glGetProgramiv(shader_id, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(shader_id, 1024, NULL, infoLog);
            fprintf(stderr, "ERROR::PROGRAM_LINKING_ERROR of type: %s\n%s", type, infoLog);
        }
    }
}
void gles2_build(gles2_controller* my_controller){
    if(my_controller->frag_shader == NULL || my_controller->ver_shader == NULL){
        perror("Have no shader to build!");
        return;
    }
    my_controller->ver_id = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(my_controller->ver_id, 1, (const char * const*)&(my_controller->ver_shader), NULL);
    glCompileShader(my_controller->ver_id);
    gles2_check_compile_error(my_controller->ver_id, "VERTEX");
    // fragment Shader
    my_controller->frag_id = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(my_controller->frag_id, 1,(const char * const*) &(my_controller->frag_shader), NULL);
    glCompileShader(my_controller->frag_id);
    gles2_check_compile_error(my_controller->frag_id, "FRAGMENT");
    // shader Program
    my_controller->pro_id = glCreateProgram();
    glAttachShader(my_controller->pro_id, my_controller->ver_id);
    glAttachShader( my_controller->pro_id, my_controller->frag_id);
    glLinkProgram( my_controller->pro_id);
    gles2_check_compile_error( my_controller->pro_id, "PROGRAM");
    // delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(my_controller->ver_id);
    glDeleteShader(my_controller->frag_id);
    glUseProgram(my_controller->pro_id);
    glFlush();

}
void gles2_compute(gles2_controller* my_controller){
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    float vertices[]={
           // positions            // texture coords
           -1.0f,  1.0f, 0.0f,  0.0f, 1.0f,  // Top left
           1.0f,  1.0f, 0.0f,  1.0f, 1.0f,     //Top right
           -1.0f, -1.0f, 0.0f,  0.0f, 0.0f,    //Bottom left
           1.0f, -1.0f, 0.0f,  1.0f, 0.0f      //Bottom right

   };
   GLuint vbo;
   glGenBuffers(1,&vbo);

   glBindBuffer(GL_ARRAY_BUFFER,vbo);

   glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
   const GLint vertex_position_size = 3;
   const GLint vertex_texcoord_size = 2;
   const GLuint vertex_position_index = 0;
   const GLuint vertex_texcoord_index = 1;
   const GLuint vertex_position_offset = 0;
   const GLuint vertex_texcoord_offset = 3;
   glBindAttribLocation(my_controller->pro_id, vertex_position_index, "pos");
   glBindAttribLocation(my_controller->pro_id, vertex_texcoord_index, "inputTexCoord");
   // position is vertex attribute 0
   glVertexAttribPointer(vertex_position_index, vertex_position_size, GL_FLOAT, GL_FALSE,
    5*sizeof(float), (void*)(vertex_position_offset));
   glEnableVertexAttribArray(vertex_position_index);
   glVertexAttribPointer(vertex_texcoord_index, vertex_texcoord_size, GL_FLOAT, GL_FALSE,
    5 * sizeof(float), (void*)(vertex_texcoord_offset*sizeof(float)));
   glEnableVertexAttribArray(vertex_texcoord_index);

   
   
   glFlush();
   //glEnable(GL_TEXTURE_2D);
   gles2_checkerror();
   
  

   
   gles2_checkerror();
   //perform compute by drawing quad
   glDrawArrays(GL_TRIANGLE_STRIP,0,4);
   glFinish();
   gles2_checkerror();


}
void gles2_setViewport(uint32_t width, uint32_t height){
    glViewport(0,0, width, height);
    gles2_checkerror();
}