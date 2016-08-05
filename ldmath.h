/*
Math functions inspired by Kazmath (https://github.com/Kazade/kazmath)
 */
#ifndef LD_MATH_HEADER
#define LD_MATH_HEADER

#include <math.h>
#include <float.h>
#include <stdlib.h>

#define LD_PI 3.1415926535897932384626f
#define LD_180_OVER_PI (180.0f / LD_PI)
#define LD_PI_OVER_180 (LD_PI / 180.0f)

/* Define how you want to export the functions */
#ifndef LD_API
#define LD_API static
#endif

LD_API float ldm_min(float a, float b) {
    return a < b ? a : b;
}

LD_API float ldm_max(float a, float b) {
    return a < b ? b : a;
}

LD_API float ldm_clamp(float x, float min, float max) {
    if (x < min) x = min;
    if (x > max) x = max;
    return x;
}

LD_API float ldm_lerp(float a, float b, float t) {
    float one_minus_t = 1 - t;
    return (a * one_minus_t) + b * t;
}

LD_API float ldm_deg_to_rad(float deg) {
    return deg * LD_PI_OVER_180;
}

LD_API float ldm_rad_to_deg(float rad) {
    return rad * LD_180_OVER_PI;
}

LD_API float ldm_floor(float in) {
    return floorf(in);
}

LD_API float ldm_ceil(float in) {
    return ceilf(in);
}

LD_API int ldm_almost_equal(float a, float b) {
    return fabs(a - b) < FLT_EPSILON;
}

LD_API int ldm_randi() {
    return rand();
}

LD_API float ldm_randf() {
    return (float) rand() / (float) RAND_MAX;
}

/*
 * Defines a vector type with n components.
 */
#define def_vecn(n) \
typedef float vec##n[n]; \
LD_API void vec##n##_add(vec##n out, vec##n const a, vec##n const b) { \
    int i; \
    for(i = 0; i < n; i++) \
        out[i] = a[i] + b[i]; \
} \
LD_API void vec##n##_sub(vec##n out, vec##n const a, vec##n const b) { \
    int i; \
    for (i = 0; i < n; i++) \
        out[i] = a[i] - b[i]; \
} \
LD_API float vec##n##_dot(const vec##n a, const vec##n b) { \
    float dot = 0; \
    int i; \
    for (i = 0; i < n; i++) \
        dot += a[i] * b[i]; \
    return dot; \
} \
LD_API void vec##n##_assign(vec##n out, const vec##n in) { \
    int i; \
    for(i = 0; i < n; i++) \
        out[i] = in[i]; \
} \
LD_API void vec##n##_scale(vec##n out, const vec##n in, float s) { \
    int i; \
    for (i = 0; i < n; i++) \
        out[i] = in[i] * s; \
} \
LD_API float vec##n##_len2(const vec##n v) { \
    return vec##n##_dot(v, v); \
} \
LD_API float vec##n##_len(const vec##n v) { \
    return sqrtf(vec##n##_dot(v, v)); \
} \
LD_API int vec##n##_norm(vec##n out, const vec##n in) { \
    float len = vec##n##_len(in); \
    float inv_len; \
    if (len == 0) \
        return 0; \
    inv_len = 1 / len; \
    vec##n##_scale(out, in, inv_len); \
    return 1; \
} \
LD_API void vec##n##_addmul(vec##n out, const vec##n in, const vec##n a, float s) { \
    int i; \
    for (i = 0; i < n; i++) \
        out[i] = in[i] + a[i] * s; \
} \
LD_API int vec##n##_equal(const vec##n a, const vec##n b) { \
    int i; \
    for (i = 0; i < n; i++) \
        if (a[i] != b[i]) \
            return 0; \
    return 1; \
} \
LD_API int vec##n##_almost_equal(const vec##n a, const vec##n b) { \
    int i; \
    for (i = 0; i < n; i++) \
        if (!ldm_almost_equal(a[i], b[i])) \
            return 0; \
    return 1; \
} \
LD_API void vec##n##_max(vec##n out, const vec##n a, const vec##n b) { \
    int i; \
    for (i = 0; i < n; i++) \
        out[i] = ldm_max(a[i], b[i]); \
} \
LD_API void vec##n##_min(vec##n out, const vec##n a, const vec##n b) { \
    int i; \
    for (i = 0; i < n; i++) \
        out[i] = ldm_min(a[i], b[i]); \
} \

def_vecn(2)
def_vecn(3)
def_vecn(4)

/* Special vector definitions */
LD_API float vec2_crossl(const vec2 a, const vec2 b) {
    return a[0] * b[1] - b[0] * a[1];
}

LD_API void vec3_cross(vec3 out, const vec3 a, const vec3 b) {
    float x = (a[1] * b[2]) - (a[2] * b[1]);
	float y = (a[2] * b[0]) - (a[0] * b[2]);
	float z = (a[0] * b[1]) - (a[1] * b[0]);
	out[0] = x;
	out[1] = y;
	out[2] = z;
}

#undef def_vecn

#define def_aabbn(n) \
typedef vec##n aabb##n[2]; \
LD_API int aabb##n##_check(aabb##n a) { \
    int i; \
    for (i = 0; i < n; i++) \
        if (a[0][i] > a[1][i]) \
            return -1; \
    return 0; \
} \
LD_API void aabb##n##_fill(aabb##n out, const vec##n min, const vec##n max) { \
    vec##n##_assign(out[0], min); \
    vec##n##_assign(out[1], max); \
} \
LD_API void aabb##n##_assign(aabb##n out, const aabb##n in) { \
    aabb##n##_fill(out, in[0], in[1]); \
} \
LD_API void aabb##n##_translate(aabb##n out, const aabb##n in, vec##n d) { \
    vec##n##_add(out[0], in[0], d); \
    vec##n##_add(out[1], in[1], d); \
} \
LD_API int aabb##n##_union(aabb##n out, const aabb##n a, const aabb##n b) { \
    vec##n##_min(out[0], a[0], b[0]); \
    vec##n##_max(out[1], a[1], b[1]); \
    return aabb##n##_check(out); \
} \
LD_API int aabb##n##_difference(aabb##n out, const aabb##n a, const aabb##n b) { \
    vec##n##_max(out[0], a[0], b[0]); \
    vec##n##_min(out[1], a[1], b[1]); \
    return aabb##n##_check(out); \
} \
LD_API int aabb##n##_contains(const aabb##n a, const vec##n p) { \
	int i; \
	for (i = 0; i < n; i++) \
		if (a[0][i] > p[i] || a[1][i] < p[i]) \
			return 0; \
	return 1; \
} \
LD_API int aabb##n##_overlaps(const aabb##n a, const aabb##n b) { \
	int i; \
	for (i = 0; i < n; i++) \
		if (a[0][i] > b[1][i] || a[1][i] < b[0][i]) \
			return 0; \
	return 1; \
} \
LD_API void aabb##n##_minkowksi(aabb##n out, const aabb##n a, const aabb##n b) { \
	int i; \
	for (i = 0; i < n; i++) { \
		out[0][i] = a[0][i] - b[1][i]; \
		out[1][i] = a[1][i] - b[0][i]; \
	} \
} \

def_aabbn(2)
def_aabbn(3)
def_aabbn(4)

#undef def_aabbn

#define def_matn(n) \
typedef float mat##n[n * n]; \
LD_API float mat##n##_get(const mat##n m, int row, int col) { \
    return m[n * col + row]; \
} \
LD_API void mat##n##_fill(mat##n out, const float * data) { \
    int i; \
    for (i = 0; i < n * n; i++) \
        out[i] = data[i]; \
} \
LD_API void mat##n##_identity(mat##n out) { \
    int i, j; \
    for (i = 0; i < n; i++) \
        for (j = 0; j < n; j++) \
            out[i + n * j] = i == j ? 1.0f : 0.0f; \
} \
LD_API void mat##n##_mul(mat##n out, const mat##n a, const mat##n b) { \
    mat##n ret; \
    float dot; \
    int c, r, x; \
    for (c = 0; c < n; c++) { \
        for (r = 0; r < n; r++) { \
            dot = 0; \
            for (x = 0; x < n; x++) \
                dot += a[n * r + x] * b[n * x + c]; \
            ret[n * r + c] = dot; \
        } \
    } \
    mat##n##_fill(out, ret); \
} \
LD_API void mat##n##_transpose(mat##n out, const mat##n m) { \
    int i, j; \
    float x, y; \
    if (out == m) { /* Swap along the diagonol */ \
        for (i = 0; i < n; i++) { \
            for (j = i + 1; j < n; j++) { \
                x = m[j * n + i]; \
                y = m[i * n + j]; \
                out[i * n + j] = x; \
                out[j * n + i] = y; \
            } \
        } \
    } else { /* Iterate through and replace all columns with rows */ \
        for (i = 0; i < n; i++) { \
            for (j = 0; j < n; j++) { \
                out[i * n + j] = m[j * n + i]; \
            } \
        } \
    } \
} \
LD_API int mat##n##_equal(const mat##n a, const mat##n b) { \
    int i; \
    for (i = 0; i < n * n; i++) \
        if (a[i] != b[i]) \
            return 0; \
    return 1; \
} \
LD_API int mat##n##_almost_equal(const mat##n a, const mat##n b) { \
    int i; \
    for (i = 0; i < n * n; i++) \
        if (!ldm_almost_equal(a[i], b[i])) \
            return 0; \
    return 1; \
}

def_matn(2)
def_matn(3)
def_matn(4)

#undef def_matn

/* Quaternions */

typedef float quat[4];

LD_API void quat_identity(quat q) {
    q[0] = q[1] = q[2] = 0.f;
    q[3] = 1.f;
}

/* vec4 macro aliases */
#define quat_scale vec4_scale
#define quat_add vec4_add
#define quat_sub vec4_sub
#define quat_dot vec4_dot

LD_API void quat_mul(quat r, quat p, quat q) {
    vec3 w;
    vec3_cross(r, p, q);
    vec3_scale(w, p, q[3]);
    vec3_add(r, r, w);
    vec3_scale(w, q, p[3]);
    vec3_add(r, r, w);
    r[3] = p[3] * q[3] - vec3_dot(p, q);
}

LD_API void quat_2mat4(const quat q, mat4 m) {

    float qx, qy, qz, qw, qx2, qy2, qz2;

    quat tmp;
    vec4_norm(tmp, q);

    qx = tmp[0];
    qy = tmp[1];
    qz = tmp[2];
    qw = tmp[3];

    qx2 = 2.0f*qx*qx;
    qy2 = 2.0f*qy*qy;
    qz2 = 2.0f*qz*qz;

    m[0] = 1.0f - qy2 - qz2;
    m[1] = 2.0f*qx*qy - 2.0f*qz*qw;
    m[2] = 2.0f*qx*qz - 2.0f*qy*qw;
    m[3] = 0.0f;

    m[4] = 2.0f*qx*qy + 2.0f*qz*qy;
    m[5] = 1.0f - qx2 - qz2;
    m[6] = 2.0f*qy*qz - 2.0f*qx*qw;
    m[7] = 0.0f;

    m[8] = 2.0f*qx*qz - 2.0f*qy*qw;
    m[9] = 2.0f*qy*qz + 2.0f*qx*qw;
    m[10]= 1.0f - qx2 - qy2;
    m[11]= 0.0f;

    m[12] = 0.0f;
    m[13] = 0.0f;
    m[14] = 0.0f;
    m[15] = 1.0f;

}

LD_API void quat_norm(quat q) {
    float scale = 1.0f / sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    quat_scale(q, q, scale);
}


LD_API void quat_conj(quat out, const quat a) {
    out[0] = -a[0];
    out[1] = -a[1];
    out[2] = -a[2];
    out[3] = a[3];
}

LD_API void quat_rot(quat q, vec3 axis, float angle) {
    int i;
    vec3 v;
    vec3_scale(v, axis, sinf(angle / 2));
    for(i = 0; i < 3; i++)
        q[i] = v[i];
    q[3] = cosf(angle / 2);
}

LD_API void quat_mul_vec3(vec3 r, quat q, vec3 v) {

    vec3 t, u;
    vec3 q_xyz;
    q_xyz[0] = q[0];
    q_xyz[1] = q[1];
    q_xyz[2] = q[2];

    vec3_cross(t, q_xyz, v);
    vec3_scale(t, t, 2);

    vec3_cross(u, q_xyz, t);
    vec3_scale(t, t, q[3]);

    vec3_add(r, v, t);
    vec3_add(r, r, u);

}

/* Transformations */

LD_API void mat4_rot_x(mat4 out, float rad) {
    /*
    | 1 0       0      0 |
    | 0 cos(A) -sin(A) 0 |
    | 0 sin(A)  cos(A) 0 |
    | 0 0       0      1 |
     */
    const float sin_ = sinf(rad), cos_ = cosf(rad);

    out[0]  = 1;
    out[1]  = 0;
    out[2]  = 0;
    out[3]  = 0;

    out[4]  = 0;
    out[5]  = cos_;
    out[6]  = -sin_;
    out[7]  = 0;

    out[8]  = 0;
    out[9]  = sin_;
    out[10] = cos_;
    out[11] = 0;

    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1;

}

LD_API void mat4_rot_y(mat4 out, float rad) {
    /*
    |  cos(A) 0 sin(A) 0 |
    |  0      1 0      0 |
    | -sin(A) 0 cos(A) 0 |
    |  0      0 0      1 |
     */
    const float sin_ = sinf(rad), cos_ = cosf(rad);

    out[0]  = cos_;
    out[1]  = 0;
    out[2]  = -sin_;
    out[3]  = 0;

    out[4]  = 0;
    out[5]  = 1;
    out[6]  = 0;
    out[7]  = 0;

    out[8]  = sin_;
    out[9]  = 0;
    out[10] = cos_;
    out[11] = 0;

    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1;

}

LD_API void mat4_rot_z(mat4 out, float rad) {
    /*
    | cos(A) -sin(A) 0 0 |
    | sin(A)  cos(A) 0 0 |
    | 0       0      1 0 |
    | 0       0      0 1 |
     */
    const float sin_ = sinf(rad), cos_ = cosf(rad);

    out[0]  = cos_;
    out[1]  = sin_;
    out[2]  = 0;
    out[3]  = 0;

    out[4]  = -sin_;
    out[5]  = cos_;
    out[6]  = 0;
    out[7]  = 0;

    out[8]  = 0;
    out[9]  = 0;
    out[10] = 1;
    out[11] = 0;

    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1;

}

LD_API void mat4_rot_ypr(mat4 out, float yaw, float pitch, float roll) {
    mat4 m1, m2;
    mat4_rot_z(m1, pitch);
    mat4_rot_x(m2, roll);
    mat4_mul(m2, m1, m2);
    mat4_rot_y(m1, yaw);
    mat4_mul(out, m1, m2);
}

LD_API void mat4_scaling(mat4 out, float x, float y, float z) {

    mat4_identity(out);

    out[0] = x;
    out[5] = y;
    out[10] = z;

}

LD_API void mat4_scaling_vec3(mat4 out, const vec3 scale) {
    mat4_scaling(out, scale[0], scale[1], scale[2]);
}

LD_API void mat4_translation(mat4 out, float x, float y, float z){

    out[0] = 1;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;

    out[4] = 0;
    out[5] = 1;
    out[6] = 0;
    out[7] = 0;

    out[8] = 0;
    out[9] = 0;
    out[10] = 1;
    out[11] = 0;

    out[12] = x;
    out[13] = y;
    out[14] = z;
    out[15] = 1;

}

LD_API void mat4_translation_vec3(mat4 out, const vec3 translate) {
    mat4_translation(out, translate[0], translate[1], translate[2]);
}

LD_API void mat4_proj_perspective(mat4 out, float fovY, float aspect, float zNear, float zFar) {

    float uh;

    if (fovY <= 0 || aspect <= 0) {
        return;
    }

    uh = 1.0f / tanf(fovY / 2.0f);

    out[0] = uh / aspect;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;

    out[4] = 0;
    out[5] = uh;
    out[6] = 0;
    out[7] = 0;

    out[8] = 0;
    out[9] = 0;
    out[10] = (zFar + zNear) / (zNear - zFar);
    out[11] = -1.0f;

    out[12] = 0;
    out[13] = 0;
    out[14] = (2.0f * zFar * zNear) / (zNear - zFar);
    out[15] = 0;

}

LD_API void mat4_proj_ortho(mat4 out,
        float left, float right,
        float bottom, float top,
        float neardist, float fardist) {

    out[0] = 2.0f / (right - left);
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;

    out[4] = 0;
    out[5] = 2.0f / (top - bottom);
    out[6] = 0;
    out[7] = 0;

    out[8] = 0;
    out[9] = 0;
    out[10] = -2.0f / (fardist - neardist);
    out[11] = 0;

    out[12] = -(right + left) / (right - left);
    out[13] = -(top + bottom) / (top - bottom);
    out[14] = -(fardist + neardist) / (fardist - neardist);
    out[15] = 1.0f;

}

LD_API void mat4_look_vec(mat4 out, const vec3 eye, const vec3 direction, const vec3 up) {

    vec3 d, s, u;

    vec3_norm(d, direction);
    vec3_scale(d, d, -1);

    vec3_cross(s, d, up);
    vec3_norm(s, s);

    vec3_cross(u, s, d);

    out[0] = s[0];
    out[1] = u[0];
    out[2] = d[0];
    out[3] = 0.0;

    out[4] = s[1];
    out[5] = u[1];
    out[6] = d[1];
    out[7] = 0.0;

    out[8] = s[2];
    out[9] = u[2];
    out[10] = d[2];
    out[11] = 0.0;

    out[12] = -vec3_dot(s, eye);
    out[13] = -vec3_dot(u, eye);
    out[14] = -vec3_dot(d, eye);
    out[15] = 1.0;

}

LD_API void mat4_look(mat4 out, float eye_x, float eye_y, float eye_z,
                 float direction_x, float direction_y, float direction_z,
                 float up_x, float up_y, float up_z) {

    vec3 eye, direction, up;

    eye[0] = eye_x;
    eye[1] = eye_y;
    eye[2] = eye_z;

    direction[0] = direction_x;
    direction[1] = direction_y;
    direction[2] = direction_z;

    up[0] = up_x;
    up[1] = up_y;
    up[2] = up_z;

    mat4_look_vec(out, eye, direction, up);

}

#endif /* end of include guard: LD_MATH_HEADER */
