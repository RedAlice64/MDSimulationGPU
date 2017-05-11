#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>


__device__ float3 float3Minus(float3 a, float3 b){
    float3 result;
    result.x = a.x-b.x;
    result.y = a.y-b.y;
    result.z = a.z-b.z;
    return result;
}

__device__ int3 int3Minus(int3 a, int3 b){
    int3 result;
    result.x = a.x-b.x;
    result.y = a.y-b.y;
    result.z = a.z-b.z;
    return result;
}

__device__ float3 float3New(float x,float y,float z){
    float3 result;
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
}


__device__ void evalForceAndEnergy(float3& force,
                                   float3 a,
                                   float3 b,
                                   float rcutsq)
{
    // compute the force divided by r in force_divr
    float force_divr;
    float3 dist = float3Minus(a,b);
    float rsq = sqrtf(dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
    if(rsq > rcutsq)return;
    float r2inv = 1.0f/rsq;
    float r6inv = r2inv * r2inv * r2inv;
    force_divr= r2inv * r6inv * (12.0f*r6inv - 6.0f);
    force.x = dist.x * force_divr;
    force.y = dist.y * force_divr;
    force.z = dist.z * force_divr;
}


__global__ void calcAngleForceCUDA(float *atom_coords,float *forces_vectors,float *angle_params,int size){
    //TODO
    // params[0]
    // r12 r32 r13  dr12 dr32 dr13


    int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * 16 + threadIdx.y;
    //printf("Hello from %f\n",index);
    if(index >= size)return;

    //read in data TODO
    float theta,sinTheta,cosTheta;
    float dpotdtheta;
    float d12,d32,d13;
    float r12[3],r32[3],r13[3],dr12[3],dr32[3],dr13[3];

    int atom_0,atom_1,atom_2,atom_3;

    atom_0 = (int)angle_params[index * 7 + 0];
    atom_1 = (int)angle_params[index * 7 + 1];
    atom_2 = (int)angle_params[index * 7 + 2];

    float *target_0,*target_1,*target_2;
    target_0 = forces_vectors + atom_0*3;
    target_1 = forces_vectors + atom_1*3;
    target_2 = forces_vectors + atom_2*3;

    r12[0] = atom_coords[(int)angle_params[index * 7] * 3 + 0] - atom_coords[(int)angle_params[index * 7 + 1] * 3 + 0];
    r12[1] = atom_coords[(int)angle_params[index * 7] * 3 + 1] - atom_coords[(int)angle_params[index * 7 + 1] * 3 + 1];
    r12[2] = atom_coords[(int)angle_params[index * 7] * 3 + 2] - atom_coords[(int)angle_params[index * 7 + 1] * 3 + 2];

    r32[0] = atom_coords[(int)angle_params[index * 7 + 2] * 3 + 0] - atom_coords[(int)angle_params[index * 7 + 1] * 3 + 0];
    r32[1] = atom_coords[(int)angle_params[index * 7 + 2] * 3 + 1] - atom_coords[(int)angle_params[index * 7 + 1] * 3 + 1];
    r32[2] = atom_coords[(int)angle_params[index * 7 + 2] * 3 + 2] - atom_coords[(int)angle_params[index * 7 + 1] * 3 + 2];

    r13[0] = atom_coords[(int)angle_params[index * 7] * 3 + 0] - atom_coords[(int)angle_params[index * 7 + 2] * 3 + 0];
    r13[1] = atom_coords[(int)angle_params[index * 7] * 3 + 1] - atom_coords[(int)angle_params[index * 7 + 2] * 3 + 1];
    r13[2] = atom_coords[(int)angle_params[index * 7] * 3 + 2] - atom_coords[(int)angle_params[index * 7 + 2] * 3 + 2];



    float n_x,n_y,n_z;
    float dot12_32 = r12[0] * r32[0] + r12[1] * r32[1] + r12[2] * r32[2];
    n_x = r12[1] * r32[2] - r12[2] * r32[1];
    n_y = r12[2] * r32[0] - r12[0] * r32[2];
    n_z = r12[0] * r32[1] - r12[1] * r32[0];
    theta = atanf(sqrtf(n_x * n_x + n_y * n_y + n_z * n_z) / dot12_32);
    sinTheta = sin(theta);
    cosTheta = cos(theta);

    //Theta0 :
    float KTheta = angle_params[index * 7 + 3];
    float Theta0 = angle_params[index * 7 + 4];

    float bradley_constant = angle_params[index * 7 + 5];
    float bradley_rest_lenth = angle_params[index * 7 + 6];

    dpotdtheta = 2.0 * KTheta * (theta - Theta0);
    d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    d32 = sqrtf(r32[0] * r32[0] + r32[1] * r32[1] + r32[2] * r32[2]);
    d13 = sqrtf(r13[0] * r13[0] + r13[1] * r13[1] + r13[2] * r13[2]);

    dr12[0] = r12[0] / d12;
    dr12[1] = r12[1] / d12;
    dr12[2] = r12[2] / d12;

    dr13[0] = r13[0] / d13;
    dr13[1] = r13[1] / d13;
    dr13[2] = r13[2] / d13;

    dr32[0] = r32[0] / d32;
    dr32[1] = r32[1] / d32;
    dr32[2] = r32[2] / d32;

    float dtheta1[3] = {
            (dr12[0] * cosTheta - dr32[0]) / (sinTheta * d12) * (-dpotdtheta) - dr13[0] * 2.0 * bradley_constant * (d13 - bradley_rest_lenth),
            (dr12[1] * cosTheta - dr32[1]) / (sinTheta * d12) * (-dpotdtheta) - dr13[1] * 2.0 * bradley_constant * (d13 - bradley_rest_lenth),
            (dr12[2] * cosTheta - dr32[2]) / (sinTheta * d12) * (-dpotdtheta) - dr13[2] * 2.0 * bradley_constant * (d13 - bradley_rest_lenth)
    };
    __threadfence();
    float dtheta3[3] = {
            (dr32[0] * cosTheta - dr12[0]) / (sinTheta * d32) * (-dpotdtheta) + dr13[0] * 2.0 * bradley_constant * (d13 - bradley_rest_lenth),
            (dr32[1] * cosTheta - dr12[1]) / (sinTheta * d32) * (-dpotdtheta) + dr13[1] * 2.0 * bradley_constant * (d13 - bradley_rest_lenth),
            (dr32[2] * cosTheta - dr12[2]) / (sinTheta * d32) * (-dpotdtheta) + dr13[2] * 2.0 * bradley_constant * (d13 - bradley_rest_lenth)
    };
    __threadfence();

    float dtheta2[3] = {
            -dtheta1[0] - dtheta3[0],
            -dtheta1[1] - dtheta3[1],
            -dtheta1[2] - dtheta3[2]
    };
    /*
    if(index < 20) {
        printf("%d:%f\t%f\t%f\t%f\t%f\t%f\t%f\n",index,
               angle_params[index*7],angle_params[index*7+1],angle_params[index*7+2],
               angle_params[index*7+3],angle_params[index*7+4],angle_params[index*7+5],angle_params[index*7+6]);

        printf("index:\t%d\tr12:%f\t%f\t%f\n", index, r12[0], r12[1], r12[2]);
        printf("index:\t%d\tr32:%f\t%f\t%f\n", index, r32[0], r32[1], r32[2]);
        printf("index:\t%d\tr13:%f\t%f\t%f\n", index, r13[0], r13[1], r13[2]);

        printf("index:\t%d\tk_theta:\t%f\tTheta0:\t%f\tBradley_Constant:\t%f\n",index,KTheta,Theta0,bradley_constant);
        printf("index:\t%d\ttheta:\t%f\n", index, theta);


        printf("index:\t%d\tdr12:%f\t%f\t%f\n", index, dr12[0], dr12[1], dr12[2]);
        printf("index:\t%d\tdr32:%f\t%f\t%f\n", index, dr32[0], dr32[1], dr32[2]);
        printf("index:\t%d\tdr13:%f\t%f\t%f\n", index, dr13[0], dr13[1], dr13[2]);

        printf("index:\t%d\tforce1:%f\t%f\t%f\n", index, dtheta1[0], dtheta1[1], dtheta1[2]);
        printf("index:\t%d\tforce2:%f\t%f\t%f\n", index, dtheta2[0], dtheta2[1], dtheta2[2]);
        printf("index:\t%d\tforce3:%f\t%f\t%f\n", index, dtheta3[0], dtheta3[1], dtheta3[2]);


    } */


    //TODO accumulate forces
 /*
    forces_vectors[(int)angle_params[index * 7] * 3 + 0] += dtheta1[0];
    __threadfence();
    forces_vectors[(int)angle_params[index * 7] * 3 + 1] += dtheta1[1];
    __threadfence();
    forces_vectors[(int)angle_params[index * 7] * 3 + 2] += dtheta1[2];
    __threadfence();

    forces_vectors[(int)angle_params[index * 7 + 1] * 3 + 0] += dtheta2[0];
    __threadfence();
    forces_vectors[(int)angle_params[index * 7 + 1] * 3 + 1] += dtheta2[1];
    __threadfence();
    forces_vectors[(int)angle_params[index * 7 + 1] * 3 + 2] += dtheta2[2];
    __threadfence();

    forces_vectors[(int)angle_params[index * 7 + 2] * 3 + 0] += dtheta3[0];
    __threadfence();
    forces_vectors[(int)angle_params[index * 7 + 2] * 3 + 1] += dtheta3[1];
    __threadfence();
    forces_vectors[(int)angle_params[index * 7 + 2] * 3 + 2] += dtheta3[2];
    __threadfence();

*/
    /*
       forces_vectors[(int)angle_params[index * 7] * 3 + 0] += dtheta1[0];
       __threadfence();
       forces_vectors[(int)angle_params[index * 7] * 3 + 1] += dtheta1[1];
       __threadfence();
       forces_vectors[(int)angle_params[index * 7] * 3 + 2] += dtheta1[2];
       __threadfence();


       forces_vectors[(int)angle_params[index * 7 +1] * 3 + 0] += dtheta2[0];
       __threadfence();
       forces_vectors[(int)angle_params[index * 7 +1] * 3 + 1] += dtheta2[1];
       __threadfence();
       forces_vectors[(int)angle_params[index * 7 +1] * 3 + 2] += dtheta2[2];
       __threadfence();

       forces_vectors[(int)angle_params[index * 7 + 2] * 3 + 0] += dtheta3[0];
       __threadfence();
       forces_vectors[(int)angle_params[index * 7 + 2] * 3 + 1] += dtheta3[1];
       __threadfence();
       forces_vectors[(int)angle_params[index * 7 + 2] * 3 + 2] += dtheta3[2];
       __threadfence();
    */
    atomicAdd(target_0,dtheta1[0]);
    atomicAdd(target_0+1,dtheta1[1]);
    atomicAdd(target_0+2,dtheta1[2]);

    atomicAdd(target_1,dtheta2[0]);
    atomicAdd(target_1+1,dtheta2[1]);
    atomicAdd(target_1+2,dtheta2[2]);

    atomicAdd(target_2,dtheta3[0]);
    atomicAdd(target_2+1,dtheta3[1]);
    atomicAdd(target_2+2,dtheta3[2]);



}


__global__ void calcBondForceCUDA(float *atom_coords,float *forces_vectors,float *bond_params,int size){
    int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * 16 + threadIdx.y;
    if(index >= size)return;
    //TODO get input
    float r[3],Kb,b0;

    int atom_0,atom_1,atom_2,atom_3;

    atom_0 = (int)bond_params[index * 4 + 0];
    atom_1 = (int)bond_params[index * 4 + 1];

    float *target_0,*target_1;
    target_0 = forces_vectors + atom_0*3;
    target_1 = forces_vectors + atom_1*3;


    r[0] = atom_coords[(int)bond_params[index * 4] * 3 + 0] - atom_coords[(int)bond_params[index * 4 + 1] * 3 + 0];
    r[1] = atom_coords[(int)bond_params[index * 4] * 3 + 1] - atom_coords[(int)bond_params[index * 4 + 1] * 3 + 1];
    r[2] = atom_coords[(int)bond_params[index * 4] * 3 + 2] - atom_coords[(int)bond_params[index * 4 + 1] * 3 + 2];

    Kb = bond_params[index * 4 + 2];
    b0 = bond_params[index * 4 + 3];

    float r_scalar = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    float dpotdr = 2.0 * Kb * (r_scalar - b0);
    r[0] *= dpotdr / r_scalar;
    r[1] *= dpotdr / r_scalar;
    r[2] *= dpotdr / r_scalar;
    //TODO accumulate forces
    /*
    forces_vectors[(int)bond_params[index * 4] * 3 + 0] += r[0];
       __threadfence();
    forces_vectors[(int)bond_params[index * 4] * 3 + 1] += r[1];
       __threadfence();
    forces_vectors[(int)bond_params[index * 4] * 3 + 2] += r[2];
       __threadfence();

    forces_vectors[(int)bond_params[index * 4 + 1] * 3 + 0] -= r[0];
       __threadfence();
    forces_vectors[(int)bond_params[index * 4 + 1] * 3 + 1] -= r[1];
       __threadfence();
    forces_vectors[(int)bond_params[index * 4 + 1] * 3 + 2] -= r[2];
       __threadfence();
    */
    atomicAdd(target_0,r[0]);
    atomicAdd(target_0+1,r[1]);
    atomicAdd(target_0+2,r[2]);

    atomicAdd(target_1,-r[0]);
    atomicAdd(target_1+1,-r[1]);
    atomicAdd(target_1+2,-r[2]);


}

__global__ void calcImproperForceCUDA(float *atom_coords,float *forces_vectors,float *improper_params,int size){
    int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * 16 + threadIdx.y;
    //printf("Hello from %f\n",index);

    if(index >= size)return;
    float r12[3],r23[3],r34[3];
    float Kchi,delta;

    int atom_0,atom_1,atom_2,atom_3;

    atom_0 = (int)improper_params[index * 6 + 0];
    atom_1 = (int)improper_params[index * 6 + 1];
    atom_2 = (int)improper_params[index * 6 + 2];
    atom_3 = (int)improper_params[index * 6 + 3];

    float *target_0,*target_1,*target_2,*target_3;
    target_0 = forces_vectors + atom_0*3;
    target_1 = forces_vectors + atom_1*3;
    target_2 = forces_vectors + atom_2*3;
    target_3 = forces_vectors + atom_3*3;

    //TODO get input
    r12[0] = atom_coords[(int)improper_params[index * 6] * 3 + 0] - atom_coords[(int)improper_params[index * 6 + 1] * 3 + 0];
    r12[1] = atom_coords[(int)improper_params[index * 6] * 3 + 1] - atom_coords[(int)improper_params[index * 6 + 1] * 3 + 1];
    r12[2] = atom_coords[(int)improper_params[index * 6] * 3 + 2] - atom_coords[(int)improper_params[index * 6 + 1] * 3 + 2];

    r34[0] = atom_coords[(int)improper_params[index * 6 + 2] * 3 + 0] - atom_coords[(int)improper_params[index * 6 + 3] * 3 + 0];
    r34[1] = atom_coords[(int)improper_params[index * 6 + 2] * 3 + 1] - atom_coords[(int)improper_params[index * 6 + 3] * 3 + 1];
    r34[2] = atom_coords[(int)improper_params[index * 6 + 2] * 3 + 2] - atom_coords[(int)improper_params[index * 6 + 3] * 3 + 2];

    r23[0] = atom_coords[(int)improper_params[index * 6 + 1] * 3 + 0] - atom_coords[(int)improper_params[index * 6 + 2] * 3 + 0];
    r23[1] = atom_coords[(int)improper_params[index * 6 + 1] * 3 + 1] - atom_coords[(int)improper_params[index * 6 + 2] * 3 + 1];
    r23[2] = atom_coords[(int)improper_params[index * 6 + 1] * 3 + 2] - atom_coords[(int)improper_params[index * 6 + 2] * 3 + 2];

    //printf("r12:%f\t%f\t%f\n",r12[0],r12[1],r12[2]);
    //printf("r23:%f\t%f\t%f\n",r23[0],r23[1],r23[2]);
    //printf("r34:%f\t%f\t%f\n",r34[0],r34[1],r34[2]);

    Kchi = improper_params[index * 6 + 4];
    delta = improper_params[index * 6 + 5];

    float a[3] = {
        r12[1] * r23[2] - r12[2] * r23[1],
        r12[2] * r23[0] - r12[0] * r23[2],
        r12[0] * r23[1] - r12[1] * r23[0]
    };
    float b[3] = {
            r23[1] * r34[2] - r23[2] * r34[1],
            r23[2] * r34[0] - r23[0] * r34[2],
            r23[0] * r34[1] - r23[1] * r34[0]
    };
    float c[3] = {
            r23[1] * a[2] - r23[2] * a[1],
            r23[2] * a[0] - r23[0] * a[2],
            r23[0] * a[1] - r23[1] * a[0]
    };
    float a_norm = a[0] * a[0] + a[1] * a[1] + a[2] * a[2],
    b_norm = b[0] * b[0] + b[1] * b[1] + b[2] * b[2],
    c_norm = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];



    a[0] /= a_norm;
    a[1] /= a_norm;
    a[2] /= a_norm;

    b[0] /= b_norm;
    b[1] /= b_norm;
    b[2] /= b_norm;

    c[0] /= c_norm;
    c[1] /= c_norm;
    c[2] /= c_norm;

    float cos_phi = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    float sin_phi = c[0] * b[0] + c[1] * b[1] + c[2] * b[2];
    float phi = -atan2f(sin_phi,cos_phi);

    float dpotdphi = 2.0 * Kchi * (phi - delta);

    float dsindC[3] = {
            (c[0] * sin_phi - b[0]) / c_norm,
            (c[1] * sin_phi - b[1]) / c_norm,
            (c[2] * sin_phi - b[2]) / c_norm
    };
    float dsindB[3] = {
            (b[0] * sin_phi - c[0]) / b_norm,
            (b[1] * sin_phi - c[1]) / b_norm,
            (b[2] * sin_phi - c[2]) / b_norm
    };
    float dcosdB[3] = {
            (b[0] * cos_phi - a[0]) / b_norm,
            (b[1] * cos_phi - a[1]) / b_norm,
            (b[2] * cos_phi - a[2]) / b_norm
    };


    if(fabsf(sin_phi) > 0.1) {
        float dcosdA[3] = {
                (a[0] * cos_phi - b[0]) / a_norm,
                (a[1] * cos_phi - b[1]) / b_norm,
                (a[2] * cos_phi - b[2]) / b_norm
        };
        float k1 = dpotdphi / sin_phi;
        float f1[3] = {
                k1 * (r23[1] * dcosdA[2] - r23[2] * dcosdA[1]),
                k1 * (r23[2] * dcosdA[0] - r23[0] * dcosdA[2]),
                k1 * (r23[0] * dcosdA[1] - r23[1] * dcosdA[0])
        };
        float f2[3] = {
                k1 * (r12[2] * dcosdA[1] - r12[1] * dcosdA[2] + r34[1] * dcosdB[2] - r34[2] * dcosdB[1]),
                k1 * (r12[0] * dcosdA[2] - r12[2] * dcosdA[0] + r34[2] * dcosdB[0] - r34[0] * dcosdB[2]),
                k1 * (r12[1] * dcosdA[0] - r12[0] * dcosdA[1] + r34[0] * dcosdB[1] - r34[1] * dcosdB[0])
        };
        float f3[3] = {
                k1 * (r23[2] * dcosdB[1] - r23[1] * dcosdB[2]),
                k1 * (r23[0] * dcosdB[2] - r23[2] * dcosdB[0]),
                k1 * (r23[1] * dcosdB[0] - r23[0] * dcosdB[1])
        };

        /*
        forces_vectors[(int)improper_params[index * 6] * 3 + 0] += f1[0];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6] * 3 + 1] += f1[1];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6] * 3 + 2] += f1[2];
        __threadfence();

        forces_vectors[(int)improper_params[index * 6 + 1] * 3 + 0] += f2[0] - f1[0];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6 + 1] * 3 + 1] += f2[1] - f1[1];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6 + 1] * 3 + 2] += f2[2] - f1[2];
        __threadfence();

        forces_vectors[(int)improper_params[index * 6 + 2] * 3 + 0] += f3[0] - f2[0];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6 + 2] * 3 + 1] += f3[1] - f2[1];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6 + 2] * 3 + 2] += f3[2] - f2[2];
        __threadfence();

        forces_vectors[(int)improper_params[index * 6 + 3] * 3 + 0] -= f3[0];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6 + 3] * 3 + 1] -= f3[1];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6 + 3] * 3 + 2] -= f3[2];
        __threadfence();
*/
        atomicAdd(target_0,f1[0]);
        atomicAdd(target_0+1,f1[1]);
        atomicAdd(target_0+2,f1[2]);

        atomicAdd(target_1,f2[0]-f1[0]);
        atomicAdd(target_1+1,f2[1]-f1[1]);
        atomicAdd(target_1+2,f2[2]-f1[2]);

        atomicAdd(target_2,f3[0]-f2[0]);
        atomicAdd(target_2+1,f3[1]-f2[1]);
        atomicAdd(target_2+2,f3[2]-f2[2]);

        atomicAdd(target_3,-f3[0]);
        atomicAdd(target_3+1,-f3[1]);
        atomicAdd(target_3+2,-f3[2]);
    }
    else {
        float k1 = -dpotdphi / cos_phi;
        float f1[3] = {
                k1 * ((r23[1] * r23[1] + r23[2] * r23[2]) * dsindC[0] - r23[0] * r23[1] * dsindC[1] -
                      r23[0] * r23[2] * dsindC[2]),
                k1 * ((r23[2] * r23[2] + r23[0] * r23[0]) * dsindC[1] - r23[1] * r23[2] * dsindC[2] -
                      r23[1] * r23[0] * dsindC[0]),
                k1 * ((r23[0] * r23[0] + r23[1] * r23[1]) * dsindC[2] - r23[2] * r23[0] * dsindC[0] -
                      r23[2] * r23[1] * dsindC[1])
        };
        float f2[3] = {
                k1 *
                (-(r23[1] * r12[1] + r23[2] * r12[2]) * dsindC[0] + (2.0 * r23[0] * r12[1] - r12[0] * r23[1]) * dsindC[1]
                 + (2.0 * r23[0] * r12[2] - r12[0] * r23[2]) * dsindC[2] + dsindB[2] * r34[1] - dsindB[1] * r34[2]),
                k1 * (-(r23[2] * r12[2] + r23[0] * r12[0]) * dsindC[1] +
                      (2.0 * r23[1] * r12[2] - r12[1] * r23[2]) * dsindC[2]
                      + (2.0 * r23[1] * r12[0] - r12[1] * r23[0]) * dsindC[0] + dsindB[0] * r34[2] -
                      dsindB[2] * r34[0]),
                k1 * (-(r23[0] * r12[0] + r23[1] * r12[1]) * dsindC[2] +
                      (2.0 * r23[2] * r12[0] - r12[2] * r23[0]) * dsindC[0]
                      + (2.0 * r23[2] * r12[1] - r12[2] * r23[1]) * dsindC[1] + dsindB[1] * r34[0] - dsindB[0] * r34[1])
        };
        float f3[3] = {
                k1 * (dsindB[1] * r23[2] - dsindB[2] * r23[1]),
                k1 * (dsindB[2] * r23[0] - dsindB[0] * r23[2]),
                k1 * (dsindB[0] * r23[1] - dsindB[1] * r23[0])
        };

/*
        forces_vectors[(int)improper_params[index * 6] * 3 + 0] += f1[0];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6] * 3 + 1] += f1[1];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6] * 3 + 2] += f1[2];
        __threadfence();

        forces_vectors[(int)improper_params[index * 6 + 1] * 3 + 0] += f2[0] - f1[0];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6 + 1] * 3 + 1] += f2[1] - f1[1];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6 + 1] * 3 + 2] += f2[2] - f1[2];
        __threadfence();

        forces_vectors[(int)improper_params[index * 6 + 2] * 3 + 0] += f3[0] - f2[0];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6 + 2] * 3 + 1] += f3[1] - f2[1];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6 + 2] * 3 + 2] += f3[2] - f2[2];
        __threadfence();

        forces_vectors[(int)improper_params[index * 6 + 3] * 3 + 0] -= f3[0];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6 + 3] * 3 + 1] -= f3[1];
        __threadfence();
        forces_vectors[(int)improper_params[index * 6 + 3] * 3 + 2] -= f3[2];
        __threadfence();*/

        atomicAdd(target_0,f1[0]);
        atomicAdd(target_0+1,f1[1]);
        atomicAdd(target_0+2,f1[2]);

        atomicAdd(target_1,f2[0]-f1[0]);
        atomicAdd(target_1+1,f2[1]-f1[1]);
        atomicAdd(target_1+2,f2[2]-f1[2]);

        atomicAdd(target_2,f3[0]-f2[0]);
        atomicAdd(target_2+1,f3[1]-f2[1]);
        atomicAdd(target_2+2,f3[2]-f2[2]);

        atomicAdd(target_3,-f3[0]);
        atomicAdd(target_3+1,-f3[1]);
        atomicAdd(target_3+2,-f3[2]);
    }
    //TODO accumulate forces


}

__device__ void extractDihedralCoords(float *atom_coords,float *dihedral_params,int index,int a_index,int b_index,float *target){
    target[0] = atom_coords[(int)dihedral_params[index * 7 + a_index] * 3 + 0] - atom_coords[(int)dihedral_params[index * 7 + b_index] * 3 + 0];
    target[1] = atom_coords[(int)dihedral_params[index * 7 + a_index] * 3 + 1] - atom_coords[(int)dihedral_params[index * 7 + b_index] * 3 + 1];
    target[2] = atom_coords[(int)dihedral_params[index * 7 + a_index] * 3 + 2] - atom_coords[(int)dihedral_params[index * 7 + b_index] * 3 + 2];
}

__global__ void calcDihedralForceCUDA(float* atom_coords,float *forces_vectors,float *dihedral_params,int size){
    int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * 16 + threadIdx.y;
    if(index >= size)return;
    int n;
    float r12[3],r23[3],r34[3];
    float Kchi,delta;

    int atom_0,atom_1,atom_2,atom_3;

    atom_0 = (int)dihedral_params[index * 7 + 0];
    atom_1 = (int)dihedral_params[index * 7 + 1];
    atom_2 = (int)dihedral_params[index * 7 + 2];
    atom_3 = (int)dihedral_params[index * 7 + 3];

    float *target_0,*target_1,*target_2,*target_3;
    target_0 = forces_vectors + atom_0*3;
    target_1 = forces_vectors + atom_1*3;
    target_2 = forces_vectors + atom_2*3;
    target_3 = forces_vectors + atom_3*3;

    extractDihedralCoords(atom_coords,dihedral_params,index,0,1,r12);

    extractDihedralCoords(atom_coords,dihedral_params,index,2,3,r34);

    extractDihedralCoords(atom_coords,dihedral_params,index,1,2,r23);

    //printf("r12:%f\t%f\t%f\n",r12[0],r12[1],r12[2]);
    //printf("r23:%f\t%f\t%f\n",r23[0],r23[1],r23[2]);
    //printf("r34:%f\t%f\t%f\n",r34[0],r34[1],r34[2]);

    Kchi = dihedral_params[index * 7 + 4];
    n = (int)dihedral_params[index * 7 + 5];
    delta = dihedral_params[index * 7 + 6];

    float a[3] = {
            r12[1] * r23[2] - r12[2] * r23[1],
            r12[2] * r23[0] - r12[0] * r23[2],
            r12[0] * r23[1] - r12[1] * r23[0]
    };
    float b[3] = {
            r23[1] * r34[2] - r23[2] * r34[1],
            r23[2] * r34[0] - r23[0] * r34[2],
            r23[0] * r34[1] - r23[1] * r34[0]
    };
    float c[3] = {
            r23[1] * a[2] - r23[2] * a[1],
            r23[2] * a[0] - r23[0] * a[2],
            r23[0] * a[1] - r23[1] * a[0]
    };
    float a_norm = a[0] * a[0] + a[1] * a[1] + a[2] * a[2],
            b_norm = b[0] * b[0] + b[1] * b[1] + b[2] * b[2],
            c_norm = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];



    a[0] /= a_norm;
    a[1] /= a_norm;
    a[2] /= a_norm;

    b[0] /= b_norm;
    b[1] /= b_norm;
    b[2] /= b_norm;

    c[0] /= c_norm;
    c[1] /= c_norm;
    c[2] /= c_norm;

    float cos_phi = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    float sin_phi = c[0] * b[0] + c[1] * b[1] + c[2] * b[2];
    float phi = -atan2f(sin_phi,cos_phi);

    float dpotdphi = n > 0 ? -(n * Kchi * sin(n * phi + delta)) : 2.0 * Kchi * (phi - delta);
    float dsindC[3] = {
            (c[0] * sin_phi - b[0]) / c_norm,
            (c[1] * sin_phi - b[1]) / c_norm,
            (c[2] * sin_phi - b[2]) / c_norm
    };
    float dsindB[3] = {
            (b[0] * sin_phi - c[0]) / b_norm,
            (b[1] * sin_phi - c[1]) / b_norm,
            (b[2] * sin_phi - c[2]) / b_norm
    };
    float dcosdB[3] = {
            (b[0] * cos_phi - a[0]) / b_norm,
            (b[1] * cos_phi - a[1]) / b_norm,
            (b[2] * cos_phi - a[2]) / b_norm
    };


    if(fabsf(sin_phi) > 0.1) {
        float dcosdA[3] = {
                (a[0] * cos_phi - b[0]) / a_norm,
                (a[1] * cos_phi - b[1]) / b_norm,
                (a[2] * cos_phi - b[2]) / b_norm
        };
        float k1 = dpotdphi / sin_phi;
        float f1[3] = {
                k1 * (r23[1] * dcosdA[2] - r23[2] * dcosdA[1]),
                k1 * (r23[2] * dcosdA[0] - r23[0] * dcosdA[2]),
                k1 * (r23[0] * dcosdA[1] - r23[1] * dcosdA[0])
        };
        float f2[3] = {
                k1 * (r12[2] * dcosdA[1] - r12[1] * dcosdA[2] + r34[1] * dcosdB[2] - r34[2] * dcosdB[1]),
                k1 * (r12[0] * dcosdA[2] - r12[2] * dcosdA[0] + r34[2] * dcosdB[0] - r34[0] * dcosdB[2]),
                k1 * (r12[1] * dcosdA[0] - r12[0] * dcosdA[1] + r34[0] * dcosdB[1] - r34[1] * dcosdB[0])
        };
        float f3[3] = {
                k1 * (r23[2] * dcosdB[1] - r23[1] * dcosdB[2]),
                k1 * (r23[0] * dcosdB[2] - r23[2] * dcosdB[0]),
                k1 * (r23[1] * dcosdB[0] - r23[0] * dcosdB[1])
        };

        /*

        forces_vectors[(int)dihedral_params[index * 7] * 3 + 0] += f1[0];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7] * 3 + 1] += f1[1];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7] * 3 + 2] += f1[2];
        __threadfence();

        forces_vectors[(int)dihedral_params[index * 7 + 1] * 3 + 0] += f2[0] - f1[0];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7 + 1] * 3 + 1] += f2[1] - f1[1];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7 + 1] * 3 + 2] += f2[2] - f1[2];
        __threadfence();

        forces_vectors[(int)dihedral_params[index * 7 + 2] * 3 + 0] += f3[0] - f2[0];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7 + 2] * 3 + 1] += f3[1] - f2[1];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7 + 2] * 3 + 2] += f3[2] - f2[2];
        __threadfence();

        forces_vectors[(int)dihedral_params[index * 7 + 3] * 3 + 0] -= f3[0];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7 + 3] * 3 + 1] -= f3[1];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7 + 3] * 3 + 2] -= f3[2];
        __threadfence();
*/

        atomicAdd(target_0,f1[0]);
        atomicAdd(target_0+1,f1[1]);
        atomicAdd(target_0+2,f1[2]);

        atomicAdd(target_1,f2[0]-f1[0]);
        atomicAdd(target_1+1,f2[1]-f1[1]);
        atomicAdd(target_1+2,f2[2]-f1[2]);

        atomicAdd(target_2,f3[0]-f2[0]);
        atomicAdd(target_2+1,f3[1]-f2[1]);
        atomicAdd(target_2+2,f3[2]-f2[2]);

        atomicAdd(target_3,-f3[0]);
        atomicAdd(target_3+1,-f3[1]);
        atomicAdd(target_3+2,-f3[2]);


    }
    else {
        float k1 = -dpotdphi / cos_phi;
        float f1[3] = {
                k1 * ((r23[1] * r23[1] + r23[2] * r23[2]) * dsindC[0] - r23[0] * r23[1] * dsindC[1] -
                      r23[0] * r23[2] * dsindC[2]),
                k1 * ((r23[2] * r23[2] + r23[0] * r23[0]) * dsindC[1] - r23[1] * r23[2] * dsindC[2] -
                      r23[1] * r23[0] * dsindC[0]),
                k1 * ((r23[0] * r23[0] + r23[1] * r23[1]) * dsindC[2] - r23[2] * r23[0] * dsindC[0] -
                      r23[2] * r23[1] * dsindC[1])
        };
        float f2[3] = {
                k1 *
                (-(r23[1] * r12[1] + r23[2] * r12[2]) * dsindC[0] + (2.0 * r23[0] * r12[1] - r12[0] * r23[1]) * dsindC[1]
                 + (2.0 * r23[0] * r12[2] - r12[0] * r23[2]) * dsindC[2] + dsindB[2] * r34[1] - dsindB[1] * r34[2]),
                k1 * (-(r23[2] * r12[2] + r23[0] * r12[0]) * dsindC[1] +
                      (2.0 * r23[1] * r12[2] - r12[1] * r23[2]) * dsindC[2]
                      + (2.0 * r23[1] * r12[0] - r12[1] * r23[0]) * dsindC[0] + dsindB[0] * r34[2] -
                      dsindB[2] * r34[0]),
                k1 * (-(r23[0] * r12[0] + r23[1] * r12[1]) * dsindC[2] +
                      (2.0 * r23[2] * r12[0] - r12[2] * r23[0]) * dsindC[0]
                      + (2.0 * r23[2] * r12[1] - r12[2] * r23[1]) * dsindC[1] + dsindB[1] * r34[0] - dsindB[0] * r34[1])
        };
        float f3[3] = {
                k1 * (dsindB[1] * r23[2] - dsindB[2] * r23[1]),
                k1 * (dsindB[2] * r23[0] - dsindB[0] * r23[2]),
                k1 * (dsindB[0] * r23[1] - dsindB[1] * r23[0])
        };
        

/*
        forces_vectors[(int)dihedral_params[index * 7] * 3 + 0] += f1[0];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7] * 3 + 1] += f1[1];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7] * 3 + 2] += f1[2];
        __threadfence();

        forces_vectors[(int)dihedral_params[index * 7 + 1] * 3 + 0] += f2[0] - f1[0];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7 + 1] * 3 + 1] += f2[1] - f1[1];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7 + 1] * 3 + 2] += f2[2] - f1[2];
        __threadfence();

        forces_vectors[(int)dihedral_params[index * 7 + 2] * 3 + 0] += f3[0] - f2[0];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7 + 2] * 3 + 1] += f3[1] - f2[1];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7 + 2] * 3 + 2] += f3[2] - f2[2];
        __threadfence();

        forces_vectors[(int)dihedral_params[index * 7 + 3] * 3 + 0] -= f3[0];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7 + 3] * 3 + 1] -= f3[1];
        __threadfence();
        forces_vectors[(int)dihedral_params[index * 7 + 3] * 3 + 2] -= f3[2];
        __threadfence();*/

        atomicAdd(target_0,f1[0]);
        atomicAdd(target_0+1,f1[1]);
        atomicAdd(target_0+2,f1[2]);

        atomicAdd(target_1,f2[0]-f1[0]);
        atomicAdd(target_1+1,f2[1]-f1[1]);
        atomicAdd(target_1+2,f2[2]-f1[2]);

        atomicAdd(target_2,f3[0]-f2[0]);
        atomicAdd(target_2+1,f3[1]-f2[1]);
        atomicAdd(target_2+2,f3[2]-f2[2]);

        atomicAdd(target_3,-f3[0]);
        atomicAdd(target_3+1,-f3[1]);
        atomicAdd(target_3+2,-f3[2]);

    }


}



__global__  void updateCoordinatesCUDA(float *atom_coords, float *atom_velocities,
                                        float timestep, int size){
    int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * 16 + threadIdx.y;
    if(index >= size)return;

    float *coord=atom_coords+index*3;

    float velocity[3] = {atom_velocities[index* 3 + 0],atom_velocities[index* 3 + 1],atom_velocities[index* 3 + 2]};

    atomicAdd(coord,velocity[0] * 0.5f * timestep);
    atomicAdd(coord+1,velocity[1] * 0.5f * timestep);
    atomicAdd(coord+2,velocity[1] * 0.5f * timestep);


/*
    atom_velocities[index * 3 + 0] = forces_vector[index * 3 + 0];
    __threadfence();
    atom_velocities[index * 3 + 1] = forces_vector[index * 3 + 1];
    __threadfence();
    atom_velocities[index * 3 + 2] = forces_vector[index * 3 + 2];
    __threadfence();

    atom_coords[index * 3 + 0] = atom_velocities[index* 3 + 0];
    __threadfence();
    atom_coords[index * 3 + 1] = atom_velocities[index* 3 + 1];
    __threadfence();
    atom_coords[index * 3 + 2] = atom_velocities[index* 3 + 2];
    __threadfence();
    */
}

__global__  void updateVelocitiesCUDA(float *atom_velocities, float *atom_masses,
                                       float *forces_vector, float timestep, int size){
    int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * 16 + threadIdx.y;
    if(index >= size)return;

    float *velocity = atom_velocities + index * 3;
    float mass = atom_masses[index];

    float force[3] = {forces_vector[index * 3 + 0],forces_vector[index * 3 + 1],forces_vector[index * 3 + 2]};

    atomicAdd(velocity,force[0]*timestep/mass);
    atomicAdd(velocity+1,force[1]*timestep/mass);
    atomicAdd(velocity+2,force[2]*timestep/mass);
}

__global__ void calcCellsCuda(int* cell_lists,
                              int cell_list_len,
                              unsigned int* cell_list_size,
                              float* atom_coords,
                              int3* max,
                              int3* min,
                              float cut_off,
                              int min_len_cell,float* force_vector){
    int cell_id = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * 16 + threadIdx.y;
    if(cell_id >= cell_list_len)return;
    int list_count = cell_list_size[cell_id];
    if(!list_count)return;

    int thread_id = threadIdx.y;


    int3 range = int3Minus(*max,*min);
    int range_index = range.x * range.y * range.z + range.y * range.z + range.z;
    int3 pos;
    pos.z = cell_id % range.z;
    pos.y = cell_id / range.z % range.y;
    pos.x = cell_id / range.y / range.y;
    for(int x = -1; x <= 1; x++){
        for(int y = -1; y <= 1; y++){
            for(int z = -1; z <= 1; z++){
                if(!(x||y||z))continue;
                int3 cur_pos;
                cur_pos.x = pos.x + x;
                cur_pos.y = pos.y + y;
                cur_pos.z = pos.z + z;
                if((cur_pos.x > range.x) || (cur_pos.x < 0) ||
                   (cur_pos.y > range.y) || (cur_pos.y < 0)||
                   (cur_pos.z > range.z) || (cur_pos.z < 0))continue;
                int tgt_cell = cur_pos.x * range.y * range.z + cur_pos.y * range.z + cur_pos.z;
                //printf("%d %d\n",cell_id,tgt_cell);
                int cell_len = cell_list_size[tgt_cell];
                //printf("%d\n",cell_len);
                for(int i = 0;i < list_count ;i++) {
                    //printf("%d\n",cell_id*min_len_cell+i);
                    int cur_particle = cell_lists[cell_id * min_len_cell + i];
                    for (int j = 0; j < cell_len; j++) {
                        int tgt_particle = cell_lists[tgt_cell*min_len_cell+j];
                        //printf("%d %d\n",cur_particle,tgt_particle);
                        float3 force;
                        float3 cur_coord = float3New(atom_coords[cur_particle * 3],atom_coords[cur_particle*3+1],atom_coords[cur_particle*3+2]);
                        float3 tgt_coord = float3New(atom_coords[tgt_particle * 3],atom_coords[tgt_particle*3+1],atom_coords[tgt_particle*3+2]);
                        evalForceAndEnergy(force,cur_coord,tgt_coord,cut_off);

                        force_vector[cur_particle * 3] += force.x;
                        force_vector[cur_particle * 3 + 1] += force.y;
                        force_vector[cur_particle * 3 + 2] += force.z;
                        //atomicAdd(force_vector + cur_particle * 3,force.x);
                        //atomicAdd(force_vector + cur_particle * 3+1, force.y);
                        //atomicAdd(force_vector + cur_particle * 3+2, force.z);

                        //atomicAdd(force_vector + tgt_particle * 3,-force.x);
                        //atomicAdd(force_vector + tgt_particle * 3+1, -force.y);
                        //atomicAdd(force_vector + tgt_particle * 3+2, -force.z);
                    }
                    int this_cell_len = cell_list_size[cell_id];
                    for(int j = i;j < this_cell_len;j++){
                        int tgt_particle = cell_lists[cell_id * min_len_cell + j];
                        float3 force;
                        float3 cur_coord ={atom_coords[cur_particle * 3],atom_coords[cur_particle*3+1],atom_coords[cur_particle*3+2]};
                        float3 tgt_coord={atom_coords[tgt_particle * 3],atom_coords[tgt_particle*3+1],atom_coords[tgt_particle*3+2]};
                        evalForceAndEnergy(force,cur_coord,tgt_coord,cut_off);

                        force_vector[cur_particle * 3] += force.x;
                        force_vector[cur_particle * 3 + 1] += force.y;
                        force_vector[cur_particle * 3 + 2] += force.z;
                        force_vector[tgt_particle * 3] += force.x;
                        force_vector[tgt_particle * 3 + 1] += force.y;
                        force_vector[tgt_particle * 3 + 2] += force.z;
                        //atomicAdd(force_vector + cur_particle * 3,force.x );
                        //atomicAdd(force_vector + cur_particle * 3+1, force.y);
                        //atomicAdd(force_vector + cur_particle * 3+2, force.z);
                        //atomicAdd(force_vector + tgt_particle * 3,-force.x);
                        //atomicAdd(force_vector + tgt_particle * 3+1, -force.y);
                        //atomicAdd(force_vector + tgt_particle * 3+2, -force.z);
                    }
                }
            }
        }

    }

}

__global__ void calcCellListCUDA(int3* cell_aligns,
                                 int *cell_lists,
                                 unsigned int* cell_list_size,
                                 int3* max,
                                 int3* min,
                                 int max_list_size,
                                 int atom_data_size){
    int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * 16 + threadIdx.y;
    if(index >= atom_data_size)return;
    int3 pos = cell_aligns[index];
    pos = int3Minus(pos,*min);
    int3 range;
    range = int3Minus(*max,*min);
    int cell_index = pos.x * range.y * range.z + pos.y * range.z + pos.z;
    int index_in_cell = atomicInc(cell_list_size + cell_index,-1 );
    cell_lists[cell_index * max_list_size + index_in_cell] = index;
}



__global__ void sortCellListCUDA(float* atom_coords,
                                 int3* cell_aligns,
                                 float3 base,
                                 int3* min,
                                 int3* max,
                                 float cut_off,
                                 int size){
    int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * 16 + threadIdx.y;
    if(index >= size)return;

    float3 atom_coord;
    atom_coord.x = atom_coords[index*3];
    atom_coord.y = atom_coords[index*3+1];
    atom_coord.z = atom_coords[index+3+2];
    int3 cell_vec;

    cell_vec.x = (int)((atom_coord.x - base.x)/cut_off);
    cell_vec.y = (int)((atom_coord.y - base.y)/cut_off);
    cell_vec.z = (int)((atom_coord.z - base.z)/cut_off);

    atomicMin(&(min->x),cell_vec.x);
    atomicMin(&(min->y),cell_vec.y);
    atomicMin(&(min->z),cell_vec.z);

    atomicMax(&(max->x),cell_vec.x);
    atomicMax(&(max->y),cell_vec.y);
    atomicMax(&(max->z),cell_vec.z);



    atomicExch(&((cell_aligns+index)->x),cell_vec.x);
    atomicExch(&((cell_aligns+index)->y),cell_vec.y);
    atomicExch(&((cell_aligns+index)->z),cell_vec.z);

}
