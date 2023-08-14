#ifndef COMMON_SHADER_H
#define COMMON_SHADER_H
#define STRINGIFY(x) #x
static const char vertex[] = STRINGIFY(
    attribute vec3 pos;
    attribute vec2 inputTexCoord;
    varying vec2 texco;
    void main() { gl_Position = vec4(pos, 1.0); 
    texco = inputTexCoord;}
);
static const char encode_decode_float_shader[] = STRINGIFY(
    precision highp float;
      const float max_float = 1.70141184e38;
    const float min_float = 1.17549435e-38;
lowp vec4 encode_float(highp float v) {
  highp float av = abs(v);
  highp vec4 c = vec4(0,0,0,0);

  //Compute exponent and mantissa
  highp float e = floor(log2(av));
  highp float m = av * pow(2.0, -e) - 1.0;

  //Unpack mantissa
  c[1] = floor(128.0 * m);
  m -= c[1] / 128.0;
  c[2] = floor(32768.0 * m);
  m -= c[2] / 32768.0;
  c[3] = floor(8388608.0 * m);
  
  //Unpack exponent
  highp float ebias = e + 127.0;
  c[0] = floor(ebias / 2.0);
  ebias -= c[0] * 2.0;
  c[1] += floor(ebias) * 128.0; 

  //Unpack sign bit
  c[0] += 128.0 * step(0.0, -v);

  //Check for 1st condition if(av < min_float) return vec4(0)
  highp vec4 res1 = vec4(step( min_float, av)*c);
  //Check for 2nd condition if(v > max_float) return vec4(127.0, 128.0, 0.0, 0.0)
  highp vec4 res2 = (1.0 - step(max_float, v))*res1 + step(max_float, v)*vec4(127.0, 128.0, 0.0, 0.0);
  //Check for 3rd condition if(v < -max_float) return  vec4(255.0, 128.0, 0.0, 0.0)
  highp vec4 res3 = (1.0 - step(v, -max_float))*res2 + step(v, -max_float)*vec4(255.0, 128.0, 0.0, 0.0);
  //Final result
  highp vec4 fin_res = res3.abgr;

  //Scale back to range
  return fin_res / 255.0;
}
float decode_float(vec4 v) {
     vec4 bits = v * 255.0;
     float sign = mix(-1.0, 1.0, step(bits[3], 128.0));
     float expo = floor(mod(bits[3] + 0.1, 128.0)) * 2.0 +
                  floor((bits[2] + 0.1) / 128.0) - 127.0;
     float sig = bits[0] +
                 bits[1] * 256.0 +
                 floor(mod(bits[2] + 0.1, 128.0)) * 256.0 * 256.0;
     return sign * (1.0 + sig / 8388607.0) * pow(2.0, expo);
   }
);
#endif