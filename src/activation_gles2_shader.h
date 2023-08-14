#include "gles2_helper.h"
#include "common_shader.h"
#ifndef ACTIVATION_SHADER_GLES2_H
#define ACTIVATION_SHADER_GLES2_H


const char frag_leaky_activate_shader[] = STRINGIFY(
    uniform sampler2D data;
    varying vec2 texco;
    void main(){
        vec4 inp_data = texture2D(data, texco);
        float a = decode_float(inp_data);
        
        float res = max(0.1*a, a);
        gl_FragColor = encode_float(res);
    }
);
const char frag_logistic_activate_shader[] = STRINGIFY(
    uniform sampler2D data;
    varying vec2 texco;
    void main(){
        vec4 inp_data = texture2D(data, texco);
        float a = decode_float(inp_data);
        //Use step instead of if to boost performance
        float res = 1.0/(1.0 + exp(-a));
        gl_FragColor = encode_float(res);
    }
);
#endif