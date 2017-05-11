//
// Created by xiuqi on 11/28/16.
//

#include <sstream>
#include <utility>

#include <tgmath.h>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>

#include <fstream>

//#include "device_launch_parameters.h"

#include "atoms.h"
#include "GPUCompute.cu"

#define EMPTY -1

inline int fact(int n)
{
    return (n==1)? 1 : n*fact(n-1);
}

Atoms::Atoms() {
    atom_data.push_back(Atom());
    forces.push_back(Vector3(0,0,0));
}

void Atoms::AddAtomFromString(std::string line) {
    atom_data.push_back(Atom(line));
    forces.push_back(Vector3(0,0,0));
}

void Atoms::AddPSFInfoToAtom(std::string line) {
    std::stringstream ss;
    ss<<line;
    int atom_id;
    float t_mass,t_charge;
    std::string t_atom_type;
    std::string temp1,temp2,temp3,temp4;
    ss>>atom_id>>temp1>>temp2>>temp3>>temp4>>t_atom_type>>t_charge>>t_mass;
    atom_data[atom_id].AddPSFData(t_mass,t_charge,t_atom_type);
}

void Atoms::AddBond(int a, int b) {
    bonds.push_back({a,b});
}

void Atoms::AddAngle(int a, int b, int c) {
    if(angle_params.find({atom_data[a].getAtom_type(),atom_data[b].getAtom_type(),atom_data[c].getAtom_type()}) != angle_params.end())angles.push_back({a,b,c});
    else if(angle_params.find({atom_data[c].getAtom_type(),atom_data[b].getAtom_type(),atom_data[a].getAtom_type()}) != angle_params.end())angles.push_back({c,b,a});
}

void Atoms::AddDihedral(int a, int b, int c, int d) {
    //if(dihedral_params.find({atom_data[a].getAtom_type(),atom_data[b].getAtom_type(),atom_data[c].getAtom_type(),atom_data[d].getAtom_type()}) != dihedral_params.end())dihedrals.push_back({a,b,c,d});
    //else if(dihedral_params.find({atom_data[d].getAtom_type(),atom_data[c].getAtom_type(),atom_data[b].getAtom_type(),atom_data[a].getAtom_type()}) != dihedral_params.end())dihedrals.push_back({d,c,b,a});

    std::vector<int> indexes={a,b,c,d};

    // Find length of string and factorial of length
    int n = 4;
    int fc = fact(n);

    // Point j to the 2nd position
    int j = 1;

    // To store position of character to be fixed next.
    // m is used as in index in s[].
    int m = 0;

    // Iterate while permutation count is
    // smaller than n! which fc
    for (int perm_c = 0; perm_c < fc; )
    {
        // Store perm as current permutation
        std::vector<int> perm = indexes;

        // Fix the first position and iterate (n-1)
        // characters upto (n-1)!
        // k is number of iterations for current first
        // character.
        int k = 0;
        while (k != fc/n)
        {
            // Swap jth value till it reaches the end position
            while (j != n-1)
            {
                // Check current permutation
                if(dihedral_params.find({atom_data[perm[0]].getAtom_type(),atom_data[perm[1]].getAtom_type(),atom_data[perm[2]].getAtom_type(),atom_data[perm[3]].getAtom_type()}) != dihedral_params.end()){
                    dihedrals.push_back({perm[0],perm[1],perm[2],perm[3]});
                    return;
                }

                // Swap perm[j] with next character
                std::swap(perm[j], perm[j+1]);

                // Increment count of permutations for this
                // cycle.
                k++;

                // Increment permutation count
                perm_c++;

                // Increment 'j' to swap with next character
                j++;
            }

            // Again point j to the 2nd position
            j = 1;
        }

        // Move to next character to be fixed in s[]
        m++;

        // If all characters have been placed at
        if (m == n)
            break;

        // Move next character to first position
        std::swap(indexes[0], indexes[m]);
    }

}

void Atoms::AddImproper(int a, int b, int c, int d) {
    //if(improper_params.find({atom_data[a].getAtom_type(),atom_data[b].getAtom_type(),atom_data[c].getAtom_type(),atom_data[d].getAtom_type()}) != improper_params.end())impropers.push_back({a,b,c,d});
    //else if(improper_params.find({atom_data[d].getAtom_type(),atom_data[c].getAtom_type(),atom_data[b].getAtom_type(),atom_data[a].getAtom_type()}) != improper_params.end())impropers.push_back({d,c,b,a});

    std::vector<int> indexes={a,b,c,d};

    // Find length of string and factorial of length
    int n = 4;
    int fc = fact(n);

    // Point j to the 2nd position
    int j = 1;

    // To store position of character to be fixed next.
    // m is used as in index in s[].
    int m = 0;

    // Iterate while permutation count is
    // smaller than n! which fc
    for (int perm_c = 0; perm_c < fc; )
    {
        // Store perm as current permutation
        std::vector<int> perm = indexes;

        // Fix the first position and iterate (n-1)
        // characters upto (n-1)!
        // k is number of iterations for current first
        // character.
        int k = 0;
        while (k != fc/n)
        {
            // Swap jth value till it reaches the end position
            while (j != n-1)
            {
                // Check current permutation
                if(improper_params.find({atom_data[perm[0]].getAtom_type(),atom_data[perm[1]].getAtom_type(),atom_data[perm[2]].getAtom_type(),atom_data[perm[3]].getAtom_type()}) !=improper_params.end()){
                    impropers.push_back({perm[0],perm[1],perm[2],perm[3]});
                    return;
                }

                // Swap perm[j] with next character
                std::swap(perm[j], perm[j+1]);

                // Increment count of permutations for this
                // cycle.
                k++;

                // Increment permutation count
                perm_c++;

                // Increment 'j' to swap with next character
                j++;
            }

            // Again point j to the 2nd position
            j = 1;
        }

        // Move to next character to be fixed in s[]
        m++;

        // If all characters have been placed at
        if (m == n)
            break;

        // Move next character to first position
        std::swap(indexes[0], indexes[m]);
    }
}

void Atoms::AddCrossTerm(int a, int b, int c, int d) {
    cross_terms.push_back({a,b,c,d});
}

void Atoms::AddBondParams(std::string a, std::string b, float Kb, float b0) {
    std::multiset<std::string> m = {a,b};
    std::vector<float> v = {Kb,b0};
    bond_params.insert(std::make_pair(m,v));
}

void Atoms::AddAngleParams(std::string a, std::string b, std::string c, float KTheta, float Theta0) {
    std::vector<std::string> m = {a,b,c};
    std::vector<float> v = {KTheta,Theta0};
    angle_params.insert(std::make_pair(m,v));
}

void Atoms::AddAngleParams(std::string a, std::string b, std::string c, float KTheta, float Theta0,float Kub,float S0) {
    std::vector<std::string> m = {a,b,c};
    std::vector<float> v = {KTheta,Theta0,Kub,S0};
    angle_params.insert(std::make_pair(m,v));
}

void Atoms::AddDihedralParams(std::string a, std::string b, std::string c, std::string d, float Kchi, int n, float delta) {
    std::vector<std::string> m = {a,b,c,d};
    std::vector<float> v = {Kchi,delta};
    std::pair<int,std::vector<float>> p = {n,v};
    dihedral_params.insert(std::make_pair(m,p));
}

void Atoms::AddImproperParams(std::string a, std::string b, std::string c, std::string d, float Kpsi, float psi0) {
    std::vector<std::string> m = {a,b,c,d};
    std::vector<float> v = {Kpsi,psi0};
    improper_params.insert(std::make_pair(m,v));
}

void Atoms::RunSingleStep(float timestep) {
    if(touched.size() < 1)touched = std::vector<bool>(atom_data.size(),false);
    this->timestep = timestep;
    AccumulateBondForce();
    AccumulateAngleForce();
    AccumulateDihedral();
    AccumulateImproper();
    //AccumulateUnbonded();
    AccumulateVelocityPosition();
    //validation();
}

void Atoms::AccumulateAngleForce() {
    int index = 0;
    for(auto angle : angles){

        std::vector<float> params = angle_params[{atom_data[angle[0]].getAtom_type(),
                                                  atom_data[angle[1]].getAtom_type(),
                                                  atom_data[angle[2]].getAtom_type()}];

        float KTheta = params[0];
        float Theta0 = params[1];

        bool is_bradley = params.size() > 2;

        float bradley_constant = !is_bradley? 0 : params[2];
        float bradley_rest_lenth = !is_bradley ? 0 : params[3];


        Vector3 r12(atom_data[angle[0]].getXYZ(),atom_data[angle[1]].getXYZ());
        Vector3 r32(atom_data[angle[2]].getXYZ(),atom_data[angle[1]].getXYZ());
        Vector3 r13(atom_data[angle[0]].getXYZ(),atom_data[angle[2]].getXYZ());

        float theta = r12.angle_with(r32);
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        //calculate dpot/dtheta
        float dpotdtheta = 2.0 * KTheta * (theta - Theta0);

        //save all distances

        float d12 = r12.norm(),d32 = r32.norm(),d13 = r13.norm();

        //calculate dirivatives of three vectors
        Vector3 dr12(r12,1 / d12);
        Vector3 dr32(r32,1 / d32);
        Vector3 dr13(r13,1 / d13);

        //calculate dtheta/dxyz
        Vector3 dtheta1(dr12);
        dtheta1 *= cosTheta;
        dtheta1 -= dr32;
        dtheta1 /= sinTheta * d12;

        Vector3 dtheta3(dr32);
        dtheta3 *= cosTheta;
        dtheta3 -= dr12;
        dtheta3 /= sinTheta * d32;

        //calculation force 1 and 3
        Vector3 force1(dtheta1,-dpotdtheta);
        Vector3 force3(dtheta3,-dpotdtheta);


        //calculate bradley force and accumulate if any
        if(is_bradley){
            Vector3 bradley_force(dr13,2.0 * bradley_constant * (d13 - bradley_rest_lenth));
            force1 -= bradley_force;
            force3 += bradley_force;
        }
        Vector3 force2(force1,-1);
        force2 -= force3;

        forces[angle[0]] += force1;
        forces[angle[1]] += force2;
        forces[angle[2]] += force3;


    }
}

void Atoms::AccumulateBondForce() {
    //TODO
    for(auto bond : bonds){

        Vector3 r12(atom_data[bond[0]].getXYZ(),atom_data[bond[1]].getXYZ());

        float r_square = r12.d_square();
        float r_scalar = sqrt(r_square);
        float Kb = bond_params[{atom_data[bond[0]].getAtom_type(),
                                atom_data[bond[1]].getAtom_type()}][0];
        float b0 = bond_params[{atom_data[bond[0]].getAtom_type(),
                                atom_data[bond[1]].getAtom_type()}][1];

        float dpotdr = 2.0 * Kb * (r_scalar - b0);

        Vector3 force(r12);
        force *= dpotdr / r_scalar;
        forces[bond[0]] += force;
        forces[bond[1]] -= force;



    }
}

void Atoms::AccumulateDihedral() {
    //TODO
    for(auto dihedral : dihedrals){



        int a1 = dihedral[0];
        int a2 = dihedral[1];
        int a3 = dihedral[2];
        int a4 = dihedral[3];

        Vector3 r12(atom_data[a1].getXYZ(),atom_data[a2].getXYZ());
        Vector3 r23(atom_data[a2].getXYZ(),atom_data[a3].getXYZ());
        Vector3 r34(atom_data[a3].getXYZ(),atom_data[a4].getXYZ());

        Vector3 a(r12,r23,false);
        Vector3 b(r23,r34,false);
        Vector3 c(r23,a,false);

        a /= a.norm();
        b /= b.norm();
        c /= c.norm();

        float cos_phi = a.dot_with(b);
        float sin_phi = c.dot_with(b);
        float phi = -atan2(sin_phi,cos_phi);

        float Kchi = dihedral_params[{atom_data[a1].getAtom_type(),
                                        atom_data[a2].getAtom_type(),
                                        atom_data[a3].getAtom_type(),
                                        atom_data[a4].getAtom_type()}].second[0];

        float delta = dihedral_params[{atom_data[a1].getAtom_type(),
                                       atom_data[a2].getAtom_type(),
                                       atom_data[a3].getAtom_type(),
                                       atom_data[a4].getAtom_type()}].second[1];

        int n = dihedral_params[{atom_data[a1].getAtom_type(),
                                 atom_data[a2].getAtom_type(),
                                 atom_data[a3].getAtom_type(),
                                 atom_data[a4].getAtom_type()}].first;

        float dpotdphi = n > 0 ? -(n * Kchi * sin(n * phi + delta)) : 2.0 * Kchi * (phi - delta);

        Vector3 f1,f2,f3;
        if(fabs(sin_phi) > 0.1){
            Vector3 dcosdA(a,cos_phi);
            dcosdA -= b;
            dcosdA /= a.norm();

            Vector3 dcosdB(b,cos_phi);
            dcosdB -= a;
            dcosdB /= b.norm();

            float k1 = dpotdphi / sin_phi;

            f1.x = k1 * (r23.y * dcosdA.z - r23.z * dcosdA.y);
            f1.y = k1 * (r23.z * dcosdA.x - r23.x * dcosdA.z);
            f1.z = k1 * (r23.x * dcosdA.y - r23.y * dcosdA.x);

            f3.x = k1 * (r23.z * dcosdB.y - r23.y * dcosdB.z);
            f3.y = k1 * (r23.x * dcosdB.z - r23.z * dcosdB.x);
            f3.z = k1 * (r23.y * dcosdB.x - r23.x * dcosdB.y);

            f2.x = k1 * (r12.z * dcosdA.y - r12.y * dcosdA.z + r34.y * dcosdB.z - r34.z * dcosdB.y);
            f2.y = k1 * (r12.x * dcosdA.z - r12.z * dcosdA.x + r34.z * dcosdB.x - r34.x * dcosdB.z);
            f2.z = k1 * (r12.y * dcosdA.x - r12.x * dcosdA.y + r34.x * dcosdB.y - r34.y * dcosdB.x);
        }
        else {
            Vector3 dsindC(c,sin_phi);
            dsindC -= b;
            dsindC /= c.norm();
            Vector3 dsindB(b,sin_phi);
            dsindB -= c;
            dsindB /= b.norm();

            float k1 = -dpotdphi/cos_phi;

            f1.x = k1 * ((r23.y * r23.y + r23.z * r23.z) * dsindC.x - r23.x * r23.y * dsindC.y - r23.x * r23.z * dsindC.z);
            f1.y = k1 * ((r23.z * r23.z + r23.x * r23.x) * dsindC.y - r23.y * r23.z * dsindC.z - r23.y * r23.x * dsindC.x);
            f1.z = k1 * ((r23.x * r23.x + r23.y * r23.y) * dsindC.z - r23.z * r23.x * dsindC.x - r23.z * r23.y * dsindC.y);

            f3 = Vector3(dsindB,r23,false);
            f3 *= k1;

            f2.x = k1 * (-(r23.y * r12.y + r23.z * r12.z) * dsindC.x +(2.0f * r23.x * r12.y - r12.x * r23.y) * dsindC.y
                         +(2.0f * r23.x * r12.z - r12.x * r23.z) * dsindC.z +dsindB.z * r34.y - dsindB.y * r34.z);
            f2.y = k1 * (-(r23.z * r12.z + r23.x * r12.x) * dsindC.y +(2.0f * r23.y * r12.z - r12.y * r23.z) * dsindC.z
                         +(2.0f * r23.y * r12.x - r12.y * r23.x) * dsindC.x +dsindB.x * r34.z - dsindB.z * r34.x);
            f2.z = k1 * (-(r23.x * r12.x + r23.y * r12.y) * dsindC.z +(2.0f * r23.z * r12.x - r12.z * r23.x) * dsindC.x
                         +(2.0f * r23.z * r12.y - r12.z * r23.y) * dsindC.y +dsindB.y * r34.x - dsindB.x * r34.y);
        }

        forces[a1] += f1;
        forces[a2] += f2;
        forces[a2] -= f1;
        forces[a3] += f3;
        forces[a3] -= f2;
        forces[a4] -= f3;





    }

}

void Atoms::AccumulateImproper() {
    for(auto improper : impropers){
        int a1 = improper[0];
        int a2 = improper[1];
        int a3 = improper[2];
        int a4 = improper[3];

        Vector3 r12(atom_data[a1].getXYZ(),atom_data[a2].getXYZ());
        Vector3 r23(atom_data[a2].getXYZ(),atom_data[a3].getXYZ());
        Vector3 r34(atom_data[a3].getXYZ(),atom_data[a4].getXYZ());

        Vector3 a(r12,r23,false);
        Vector3 b(r23,r34,false);
        Vector3 c(r23,a,false);

        a /= a.norm();
        b /= b.norm();
        c /= c.norm();

        float cos_phi = a.dot_with(b);
        float sin_phi = c.dot_with(b);
        float phi = -atan2(sin_phi,cos_phi);//TODO modify previous atan()

        float Kchi = improper_params[{atom_data[a1].getAtom_type(),
                                      atom_data[a2].getAtom_type(),
                                      atom_data[a3].getAtom_type(),
                                      atom_data[a4].getAtom_type()}][0];

        float delta = improper_params[{atom_data[a1].getAtom_type(),
                                       atom_data[a2].getAtom_type(),
                                       atom_data[a3].getAtom_type(),
                                       atom_data[a4].getAtom_type()}][1];

        float dpotdphi = 2.0f * Kchi * (phi - delta);

        Vector3 dsindC(c,sin_phi);// TODO this part needs check
        dsindC -= b;
        dsindC /= c.norm();
        Vector3 dsindB(b,sin_phi);
        dsindB -= c;
        dsindB /= b.norm();

        //float k1 = -dpotdphi/cos_phi;

        Vector3 f1,f2,f3;
        if(fabs(sin_phi) > 0.1){
            Vector3 dcosdA(a,cos_phi);
            dcosdA -= b;
            dcosdA /= a.norm();

            Vector3 dcosdB(b,cos_phi);
            dcosdB -= a;
            dcosdB /= b.norm();

            float k1 = dpotdphi / sin_phi;

            f1.x = k1 * (r23.y * dcosdA.z - r23.z * dcosdA.y);
            f1.y = k1 * (r23.z * dcosdA.x - r23.x * dcosdA.z);
            f1.z = k1 * (r23.x * dcosdA.y - r23.y * dcosdA.x);

            f3.x = k1 * (r23.z * dcosdB.y - r23.y * dcosdB.z);
            f3.y = k1 * (r23.x * dcosdB.z - r23.z * dcosdB.x);
            f3.z = k1 * (r23.y * dcosdB.x - r23.x * dcosdB.y);

            f2.x = k1 * (r12.z * dcosdA.y - r12.y * dcosdA.z + r34.y * dcosdB.z - r34.z * dcosdB.y);
            f2.y = k1 * (r12.x * dcosdA.z - r12.z * dcosdA.x + r34.z * dcosdB.x - r34.x * dcosdB.z);
            f2.z = k1 * (r12.y * dcosdA.x - r12.x * dcosdA.y + r34.x * dcosdB.y - r34.y * dcosdB.x);
        }
        else {
            Vector3 dsindC(c,sin_phi);
            dsindC -= b;
            dsindC /= c.norm();
            //Vector3 dsindB(b,sin_phi);
            dsindB -= c;
            dsindB /= b.norm();

            float k1 = -dpotdphi/cos_phi;

            f1.x = k1 * ((r23.y * r23.y + r23.z * r23.z) * dsindC.x - r23.x * r23.y * dsindC.y - r23.x * r23.z * dsindC.z);
            f1.y = k1 * ((r23.z * r23.z + r23.x * r23.x) * dsindC.y - r23.y * r23.z * dsindC.z - r23.y * r23.x * dsindC.x);
            f1.z = k1 * ((r23.x * r23.x + r23.y * r23.y) * dsindC.z - r23.z * r23.x * dsindC.x - r23.z * r23.y * dsindC.y);

            f3 = Vector3(dsindB,r23,false);
            f3 *= k1;

            f2.x = k1 * (-(r23.y * r12.y + r23.z * r12.z) * dsindC.x +(2.0f * r23.x * r12.y - r12.x * r23.y) * dsindC.y
                         +(2.0f * r23.x * r12.z - r12.x * r23.z) * dsindC.z +dsindB.z * r34.y - dsindB.y * r34.z);
            f2.y = k1 * (-(r23.z * r12.z + r23.x * r12.x) * dsindC.y +(2.0f * r23.y * r12.z - r12.y * r23.z) * dsindC.z
                         +(2.0f * r23.y * r12.x - r12.y * r23.x) * dsindC.x +dsindB.x * r34.z - dsindB.z * r34.x);
            f2.z = k1 * (-(r23.x * r12.x + r23.y * r12.y) * dsindC.z +(2.0f * r23.z * r12.x - r12.z * r23.x) * dsindC.x
                         +(2.0f * r23.z * r12.y - r12.z * r23.y) * dsindC.y +dsindB.y * r34.x - dsindB.x * r34.y);
        }


        forces[a1] += f1;
        forces[a2] += f2;
        forces[a2] -= f1;
        forces[a3] += f3;
        forces[a3] -= f2;
        forces[a4] -= f3;
        touched[a1] = touched[a2] = touched[a3] = touched[a4] = true;
    }
}

void Atoms::AccumulateUnbonded() {
    //
    //for(int i = 0;i < atom_data.size();i++){
        /*
        for(int j = i+1;j < atom_data.size();j++){
            Vector3 d(atom_data[i].getXYZ(),atom_data[j].getXYZ());
            float r2_i = 1 / d.d_square(),r6_i = r2_i * r2_i * r2_i;
            float ff = r2_i * r6_i * (r6_i - 0.5);
            d *= 48*ff;
            forces[i] += d;
            forces[j] -= d;
        }*/

    int lc[3] = {10,10,10};



    int list_head[lc[0]*lc[1]*lc[2]];
    int list_next[atom_data.size()];
    Vector3 min(0,0,0),max(0,0,0);

        for(std::size_t i = 1;i < atom_data.size();i++)
        {
            list_next[i] = -1;
            if(atom_data[i].getX() < min.x)min.x = atom_data[i].getX();
            if(atom_data[i].getX() > max.x)max.x = atom_data[i].getX();
            if(atom_data[i].getY() < min.y)min.y = atom_data[i].getY();
            if(atom_data[i].getY() > max.y)max.y = atom_data[i].getY();
            if(atom_data[i].getZ() < min.z)min.z = atom_data[i].getZ();
            if(atom_data[i].getZ() > max.z)max.z = atom_data[i].getZ();
        }
    min -= 0.5f;
    max += 0.5f;

        double rc[3] = {(max.x - min.x)/lc[0],(max.y - min.y)/lc[1],(max.z - min.z)/lc[2]};
        for(std::size_t i = 0;i < lc[0] * lc[1] * lc[2];i++)
        {
            list_head[i] = -1;
        }

        for(std::size_t i = 1;i < atom_data.size();i++)
        {
            int block_number[3] = {(atom_data[i].getX() - min.getX()) / rc[0],
                                   (atom_data[i].getY() - min.getY()) / rc[1],
                                   (atom_data[i].getZ() - min.getZ()) / rc[2]};
            int linear_number = block_number[0] * lc[1] * lc[2] +
                                block_number[1] * lc[2] +
                                block_number[2];
            if(linear_number == 1007)
                std::cout<<linear_number<<std::endl;
            list_next[i] = list_head[linear_number];
            list_head[linear_number] = i;

        }
        for(int i = 0;i < lc[0];i++)
            for(int j = 0;j < lc[1];j++)
                for(int k = 0;k < lc[2];k++)
                {
                    int linear_number = i * lc[1] * lc[2] +
                                        j * lc[2] +
                                        k;
                    for(int ii = i - 1;ii < i + 2;ii++)
                        for(int jj = j - 1;jj < j + 2;jj++)
                            for(int kk = k - 1;kk < k + 2;kk++)
                            {


                                int linear_number1 =
                                        ((ii + lc[0]) % lc[0]) * lc[0] * lc[1] +
                                        ((jj + lc[1]) % lc[1]) * lc[1] +
                                        ((kk + lc[2]) % lc[2]);

                                int p,q,r;
                                p = list_head[linear_number];
                                while(p != EMPTY)
                                {
                                    q = list_head[linear_number1];
                                    while(q != EMPTY)
                                    {
                                        if(p < q)
                                        {
                                            Vector3 pp = atom_data[p].getXYZ();
                                            Vector3 qq = atom_data[q].getXYZ();
                                            Vector3 rpq = Vector3(pp,qq);
                                            double r2 = rpq.d_square();
                                            if(r2 < 4.0) {
                                                double r2_i = 1 / r2, r6_i = r2_i * r2_i * r2_i;
                                                double ff = r2_i * r6_i * (r6_i - 0.5);

                                                forces[p] += Vector3(rpq, ff * 48);
                                                forces[q] -= Vector3( rpq, ff * 48);
                                                //en += 4 * (r6_i * (r6_i - 1));

                                                //std::cout<<p<<":"<<q<<" : "<<rpq.x<<" "<<rpq.y<<" "<<rpq.z<<std::endl;
                                            }
                                        }
                                        q = list_next[q];
                                    }
                                    p = list_next[p];
                                }

                            }

                }
    //}
}

void Atoms::AccumulateVelocityPosition() {
    //TODO
    for(int i = 0;i < atom_data.size();i++){
        atom_data[i].intergrateCoordinate(timestep);
        atom_data[i].intergrateVelocity(forces[i],timestep);
        atom_data[i].intergrateCoordinate(timestep);
    }
}

void Atoms::initializeGPUMemoryFormat() {
    atom_coords = (float*) malloc(atom_data.size() * 3 * sizeof(float));
    atom_masses_cpu = (float*) malloc(atom_data.size() * sizeof(float));
    velocities_cpu = (float*) malloc(atom_data.size() * 3 * sizeof(float));
    forces_cpu = (float*) malloc(atom_data.size() * 3 * sizeof(float));
    bond_param_cpu = (float*) malloc(bonds.size() * 4 * sizeof(float));
    angle_param_cpu = (float*) malloc(angles.size() * 7 * sizeof(float));
    dihedral_param_cpu = (float*) malloc(dihedrals.size() * 7 * sizeof(float));
    improper_param_cpu = (float*) malloc(impropers.size() * 6 * sizeof(float));

    //cell_aligns = (int3*) malloc(atom_data.size() * sizeof(int3));
    //cell_lists = (int*) malloc(atom_data.size() * 20 * sizeof(int));
    //cell_list_size = (unsigned int*) malloc(atom_data.size() * sizeof(unsigned int));

    memset(forces_cpu,0,atom_data.size() * 3 * sizeof(float));
    memset(velocities_cpu,0,atom_data.size() * 3 * sizeof(float));

    min_cpu = new int3;
    max_cpu = new int3;

    min_cpu->x = 2147483647;
    min_cpu->y = 2147483647;
    min_cpu->z = 2147483647;


    //memset(cell_aligns,-1,atom_data.size() * sizeof(int3));
    //memset(cell_lists,-1,atom_data.size() * 20 * sizeof(int));
    //memset(cell_list_size,0,atom_data.size() * sizeof(int));



    for(int i = 0;i < atom_data.size();i++){
        atom_coords[i * 3 + 0] = atom_data[i].getX();
        atom_coords[i * 3 + 1] = atom_data[i].getY();
        atom_coords[i * 3 + 2] = atom_data[i].getZ();
        atom_masses_cpu[i] = atom_data[i].mass;
    }
    for(int i = 0;i < bonds.size();i++){
        bond_param_cpu[i * 4 + 0] = bonds[i][0];
        bond_param_cpu[i * 4 + 1] = bonds[i][1];
        bond_param_cpu[i * 4 + 2] = bond_params[{atom_data[bonds[i][0]].getAtom_type(),
                                                atom_data[bonds[i][1]].getAtom_type()}][0];
        bond_param_cpu[i * 4 + 3] = bond_params[{atom_data[bonds[i][0]].getAtom_type(),
                                                 atom_data[bonds[i][1]].getAtom_type()}][1];
    }
    for(int i = 0;i < angles.size();i++){
        angle_param_cpu[i * 7 + 0] = angles[i][0];
        angle_param_cpu[i * 7 + 1] = angles[i][1];
        angle_param_cpu[i * 7 + 2] = angles[i][2];
        std::vector<float> angle_param = angle_params[{atom_data[angles[i][0]].getAtom_type(),
                                                        atom_data[angles[i][1]].getAtom_type(),
                                                        atom_data[angles[i][2]].getAtom_type()}];
        angle_param_cpu[i * 7 + 3] = angle_param[0];
        angle_param_cpu[i * 7 + 4] = angle_param[1];
        angle_param_cpu[i * 7 + 5] = angle_param.size() > 2 ? angle_param[2]:0;
        angle_param_cpu[i * 7 + 6] = angle_param.size() > 2 ? angle_param[3]:0;


    }
    for(int i = 0;i < dihedrals.size();i++){
        dihedral_param_cpu[i * 7 + 0] = dihedrals[i][0];
        dihedral_param_cpu[i * 7 + 1] = dihedrals[i][1];
        dihedral_param_cpu[i * 7 + 2] = dihedrals[i][2];
        dihedral_param_cpu[i * 7 + 3] = dihedrals[i][3];
        std::pair<int,std::vector<float>> dihedral_param = dihedral_params[{atom_data[dihedrals[i][0]].getAtom_type(),
                                                                            atom_data[dihedrals[i][1]].getAtom_type(),
                                                                            atom_data[dihedrals[i][2]].getAtom_type(),
                                                                            atom_data[dihedrals[i][3]].getAtom_type()}];
        dihedral_param_cpu[i * 7 + 4] = dihedral_param.second[0];
        dihedral_param_cpu[i * 7 + 5] = dihedral_param.first;
        dihedral_param_cpu[i * 7 + 6] = dihedral_param.second[1];

    }

    for(int i = 0;i < impropers.size();i++){
        improper_param_cpu[i * 6 + 0] = impropers[i][0];
        improper_param_cpu[i * 6 + 1] = impropers[i][1];
        improper_param_cpu[i * 6 + 2] = impropers[i][2];
        improper_param_cpu[i * 6 + 3] = impropers[i][3];
        std::vector<float> improper_param = improper_params[{atom_data[impropers[i][0]].getAtom_type(),
                                                             atom_data[impropers[i][1]].getAtom_type(),
                                                             atom_data[impropers[i][2]].getAtom_type(),
                                                             atom_data[impropers[i][3]].getAtom_type()}
                                                        ];
        improper_param_cpu[i * 6 + 4] = improper_param[0];
        improper_param_cpu[i * 6 + 5] = improper_param[1];

    }



}

void Atoms::initializeGPUDevice() {
    cudaError_t error;

    error = cudaSetDevice(0);
    if(error != cudaSuccess){
        std::cout<<"set device fail"<<std::endl;
        exit(1);
    }

    error = cudaMalloc((void **)&atom_coords_gpu,atom_data.size() * 3 * sizeof(float));
    if(error != cudaSuccess){
        std::cout<<"malloc cords fail"<<std::endl;
        exit(1);
    }

    error = cudaMalloc((void **)&atom_masses_gpu,atom_data.size() * sizeof(float));
    if(error != cudaSuccess){
        std::cout<<"malloc masses fail"<<std::endl;
        exit(1);
    }
    error = cudaMalloc((void **)&forces_gpu,atom_data.size() * 3 * sizeof(float));
    if(error != cudaSuccess){
        std::cout<<"malloc force fail"<<std::endl;
        exit(1);
    }
    error = cudaMalloc((void **)&velocities_gpu,atom_data.size() * 3 * sizeof(float));
    if(error != cudaSuccess){
        std::cout<<"malloc velocities fail"<<std::endl;
        exit(1);
    }
    error = cudaMalloc((void **)&bond_param_gpu,bonds.size() * 4 * sizeof(float));
    if(error != cudaSuccess){
        std::cout<<"malloc bonds fail"<<std::endl;
        exit(1);
    }
    error = cudaMalloc((void **)&angle_param_gpu,angles.size() * 7 * sizeof(float));
    if(error != cudaSuccess){
        std::cout<<"malloc angles fail"<<std::endl;
        exit(1);
    }
    error = cudaMalloc((void **)&dihedral_param_gpu,dihedrals.size() * 7 * sizeof(float));
    if(error != cudaSuccess){
        std::cout<<"malloc dihedrals fail"<<std::endl;
        exit(1);
    }
    error = cudaMalloc((void **)&improper_param_gpu,impropers.size() * 6 * sizeof(float));
    if(error != cudaSuccess){
        std::cout<<"malloc impropers fail"<<std::endl;
        exit(1);
    }

    error = cudaMalloc((void **)&cell_aligns_gpu,atom_data.size() * sizeof(int3));
    if(error != cudaSuccess){
        std::cout<<"malloc cell aligns fail"<<std::endl;
        exit(1);
    }

    error = cudaMalloc((void **)&cell_lists_gpu,atom_data.size() * 30 * sizeof(int));
    if(error != cudaSuccess){
        std::cout<<"malloc cell lists fail"<<std::endl;
        exit(1);
    }

    error = cudaMalloc((void **)&cell_list_size_gpu,atom_data.size() * sizeof(unsigned int));
    if(error != cudaSuccess){
        std::cout<<"malloc list sizes fail"<<std::endl;
        exit(1);
    }

    error = cudaMalloc((void **)&max_gpu,sizeof(int3));
    if(error != cudaSuccess){
        std::cout<<"malloc max gpu fail"<<std::endl;
        exit(1);
    }

    error = cudaMalloc((void **)&min_gpu,sizeof(int3));
    if(error != cudaSuccess){
        std::cout<<"malloc min gpu fail"<<std::endl;
        exit(1);
    }

    error = cudaMemcpy(atom_coords_gpu,atom_coords,atom_data.size() * 3 * sizeof(float),cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        std::cout<<"copy coords fail"<<std::endl;
        exit(1);
    }
    error = cudaMemcpy(atom_masses_gpu,atom_masses_cpu,atom_data.size() * sizeof(float),cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        std::cout<<"copy masses fail"<<std::endl;
        exit(1);
    }
    error = cudaMemcpy(velocities_gpu,velocities_cpu,atom_data.size() * 3 * sizeof(float),cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        std::cout<<"copy velocities fail"<<std::endl;
        exit(1);
    }
    error = cudaMemcpy(forces_gpu,forces_cpu,atom_data.size() * 3 * sizeof(float),cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        std::cout<<"copy force fail"<<std::endl;
        exit(1);
    }
    error = cudaMemcpy(bond_param_gpu,bond_param_cpu,bonds.size() * 4 * sizeof(float),cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        std::cout<<"copy bonds fail"<<std::endl;
        exit(1);
    }
    error = cudaMemcpy(angle_param_gpu,angle_param_cpu,angles.size() * 7 * sizeof(float),cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        std::cout<<"copy angles fail"<<std::endl;
        exit(1);
    }
    error = cudaMemcpy(dihedral_param_gpu,dihedral_param_cpu,dihedrals.size() * 7 * sizeof(float),cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        std::cout<<"copy dihedrals fail"<<std::endl;
        exit(1);
    }
    error = cudaMemcpy(improper_param_gpu,improper_param_cpu,impropers.size() * 6 * sizeof(float),cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        std::cout<<"copy impropers fail"<<std::endl;
        exit(1);
    }
    error = cudaMemset(cell_aligns_gpu,-1,atom_data.size() * sizeof(int3));
    if(error != cudaSuccess){
        std::cout<<"copy cell aligns fail"<<std::endl;
        exit(1);
    }
    error = cudaMemset(cell_lists_gpu,0,atom_data.size() * 20 * sizeof(int));
    if(error != cudaSuccess){
        std::cout<<"set cell lists fail"<<std::endl;
        exit(1);
    }
    error = cudaMemset(cell_list_size_gpu,0,atom_data.size() * sizeof(unsigned int));
    if(error != cudaSuccess){
        std::cout<<"copy cell size fail"<<std::endl;
        exit(1);
    }

    error = cudaMemset(max_gpu,-1,sizeof(int3));
    if(error != cudaSuccess){
        std::cout<<"set max vector fail"<<std::endl;
        exit(1);
    }

    error = cudaMemcpy(min_gpu,min_cpu,sizeof(int3),cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        std::cout<<"set min vector fail"<<std::endl;
        exit(1);
    }




}

void Atoms::singleGPUStep() {
    dim3 thread(16,16,1);
    dim3 grid(1,1,1);
    grid.x = bonds.size() / 256;
    if(bonds.size() % 256)grid.x++;

    cudaError_t error;// = cudaDeviceSynchronize();


    calcBondForceCUDA<<<grid,thread>>>(atom_coords_gpu,forces_gpu,bond_param_gpu,bonds.size());
    error = cudaDeviceSynchronize();
    if(error != cudaSuccess){
        std::cout<<"bond calc error"<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }


    grid.x = angles.size() / 256;
    if(angles.size() % 256)grid.x++;
    calcAngleForceCUDA<<<grid,thread>>>(atom_coords_gpu,forces_gpu,angle_param_gpu,angles.size());
    error = cudaDeviceSynchronize();
    if(error != cudaSuccess){
        std::cout<<"angle calc error"<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }


    grid.x = dihedrals.size() / 256;
    if(dihedrals.size() % 256)grid.x++;
    calcDihedralForceCUDA<<<grid,thread>>>(atom_coords_gpu,forces_gpu,dihedral_param_gpu,dihedrals.size());
    error = cudaDeviceSynchronize();
    if(error != cudaSuccess){
        std::cout<<"dihedral calc error"<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }


    grid.x = impropers.size() / 256;
    if(impropers.size() % 256)grid.x++;
    calcImproperForceCUDA<<<grid,thread>>>(atom_coords_gpu,forces_gpu,improper_param_gpu,impropers.size());
    error = cudaDeviceSynchronize();
    if(error != cudaSuccess){
        std::cout<<"improper calc error"<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    float3 base;
    base.x = -10;
    base.y = -20;
    base.z = -50;

    grid.x = atom_data.size() / 256;
    if(atom_data.size() % 256)grid.x++;
    sortCellListCUDA<<<grid,thread>>>(atom_coords_gpu,cell_aligns_gpu,base,min_gpu,max_gpu,cut_off,atom_data.size());
    error = cudaDeviceSynchronize();
    if(error != cudaSuccess){
        std::cout<<"sort  error"<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }




    calcCellListCUDA<<<grid,thread>>>(cell_aligns_gpu,cell_lists_gpu,cell_list_size_gpu,max_gpu,min_gpu,50,atom_data.size());
    error = cudaDeviceSynchronize();
    if(error != cudaSuccess){
        std::cout<<"calc  error "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }


    error = cudaMemcpy(max_cpu,max_gpu,sizeof(int3),cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
        std::cout<<"max copy back error "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    error = cudaMemcpy(min_cpu,min_gpu,sizeof(int3),cudaMemcpyDeviceToHost);
    if(error != cudaSuccess){
        std::cout<<"min copy back error "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    int3 range;
    range.x = max_cpu->x - min_cpu->x;
    range.y = max_cpu->y - min_cpu->y;
    range.z = max_cpu->z - min_cpu->z;
    int range_id = range.x * range.y * range.z;

    grid.x = range_id/256;
    if(range_id % 256)grid.x++;
    calcCellsCuda<<<grid,thread>>>(cell_lists_gpu,range_id,cell_list_size_gpu,atom_coords_gpu,max_gpu,min_gpu,cut_off,50,forces_gpu);
    error = cudaDeviceSynchronize();
    if(error !=cudaSuccess){
        std::cout<<"non bonded error "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    grid.x = range_id / 256;
    if(range_id % 256)grid.x++;
    updateCoordinatesCUDA<<<grid,thread>>>(atom_coords_gpu,velocities_gpu,timestep,atom_data.size());
    error = cudaDeviceSynchronize();
    updateVelocitiesCUDA<<<grid,thread>>>(velocities_gpu,atom_masses_gpu,forces_gpu,timestep,atom_data.size());
    error = cudaDeviceSynchronize();
    updateCoordinatesCUDA<<<grid,thread>>>(atom_coords_gpu,velocities_gpu,timestep,atom_data.size());
    error = cudaDeviceSynchronize();
    if(error != cudaSuccess){
        std::cout<<"accumulate calc error"<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

}

void Atoms::PrepareGPU(float step) {
    timestep = step;
    initializeGPUMemoryFormat();
    initializeGPUDevice();
}

void Atoms::RunSingleStepGPU() {
    singleGPUStep();
}

void Atoms::validation() {
    std::ofstream fout;
    fout.open("result");

    cudaError_t error;
    error = cudaMemcpy(atom_coords,atom_coords_gpu,atom_data.size() * 3 * sizeof(float),cudaMemcpyDeviceToHost);
    //error = cudaMemcpy(atom_coords_gpu,atom_coords,atom_data.size() * 3 * sizeof(float),cudaMemcpyHostToDevice);
    //error = cudaMemcpy(angle_param_cpu,angle_param_gpu,angles.size() * 7 * sizeof(float),cudaMemcpyDeviceToHost);

    if(error != cudaSuccess){
        exit(2);
    }
    for(int i = 0;i < atom_data.size();i++){
        //if(touched[i]) {
            //fout << atom_data[i].getX() << " " << atom_data[i].getY() << " " << atom_data[i].getZ() << std::endl;

            fout << atom_coords[i * 3 + 0] << " " << atom_coords[i * 3 + 1] << " " << atom_coords[i * 3 + 2]
                 << std::endl;
        //}
    }
    //for(int i = 0;i < bonds.size();i++){
    //    std::cout<<i<<" "<<bond_param_cpu[i * 4]<<" "<<bond_param_cpu[i*4 + 1]<<" "<<bond_param_cpu[i* 4 + 2] <<" "<<bond_param_cpu[i*4 + 3]<<std::endl;
    //}
}