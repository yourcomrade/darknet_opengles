#include "activation_layer.h"
#include "utils.h"
#include "dark_cuda.h"
#include "blas.h"
#include "gemm.h"
#include "gles2_helper.h"
#include "activation_gles2_shader.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = { (LAYER_TYPE)0 };
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = (float*)xcalloc(batch * inputs, sizeof(float));
    l.delta = (float*)xcalloc(batch * inputs, sizeof(float));


    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
// #ifdef GLES2
//     printf("Make GLES2 activation layer\n");
//     l.forward = forward_activation_layer_gles2;
// #endif
#ifdef GPU
    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    l.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}

void forward_activation_layer(layer l, network_state state)
{
    printf("Use forward_activaion_layer\n");
    copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_activation_layer(layer l, network_state state)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    copy_cpu(l.outputs*l.batch, l.delta, 1, state.delta, 1);
}

#ifdef GPU

void forward_activation_layer_gpu(layer l, network_state state)
{
    copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_gpu(layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, state.delta, 1);
}
#endif
/*
#ifdef GLES2
char * choose_activate_shader(ACTIVATION a){
   char *res = NULL;
    switch(a){
        case LEAKY:
            res = (char*)(calloc(sizeof(encode_decode_float_shader) + sizeof(frag_leaky_activate_shader), 
            sizeof(char)));
            break;
        
        case LOGISTIC:
            res = (char*)(calloc(sizeof(encode_decode_float_shader) + sizeof(frag_logistic_activate_shader), 
            sizeof(char)));
            break;
        default:
            return NULL;
    }
    if( res == NULL){
        perror("Cannot allocate memory for fragment shader activation!");
        abort();
    }
    else{
        strcat(res, encode_decode_float_shader);
        switch (a)
        {
        case LEAKY:
            strcat(res, frag_leaky_activate_shader);
            break;
        case LOGISTIC:
            strcat(res, frag_logistic_activate_shader);
        
        default:
            break;
        }
        return res;
    }
}

void forward_activation_layer_gles2(layer l, network_state state)
{
    //copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    //activate_array_gles2(l.output, l.outputs*l.batch, l.activation);
    printf("USE forward activation gles\n");
    if(l.activation == LINEAR){
        copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    }
    else{
        gles2_data* data = gles2_make_farr(state.input, l.outputs*l.batch); 
        
        gles2_data* res = gles2_make_farr(NULL, l.outputs*l.batch); 
        gles2_make_surface(my_sp, res->textSize, res->textSize);
        gles2_push_farr(my_con, res, NULL, false);

        my_con->ver_shader = vertex;
        my_con->frag_shader = choose_activate_shader(l.activation);
        
        gles2_build(my_con);
        gles2_push_farr(my_con, data, "data", true);

        gles2_make_fbo(my_con, res);

        gles2_setViewport(res->textSize, res->textSize);
        gles2_compute(my_con);
        gles2_pull_farr(l.output, l.outputs*l.batch, res);

        gles2_free_dev_farr(data);
   
        gles2_free_dev_farr(res);
        gles2_destroy_fbo(my_con);
        gles2_destroy_surface(my_sp);
    
    }
}
#endif
*/