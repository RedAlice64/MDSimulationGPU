//
// Created by xiuqi on 11/28/16.
//

#ifndef READ_PDB_ATOMS_H
#define READ_PDB_ATOMS_H

#include <vector>
#include <string>
#include "atom.h"
#include <set>
#include <map>
#include <utility>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>


class Atoms {
private:
    std::vector<Atom> atom_data;
    std::vector<std::vector<int>> bonds;
    std::vector<std::vector<int>> angles;
    std::vector<std::vector<int>> dihedrals;
    std::vector<std::vector<int>> impropers;
    std::vector<std::vector<int>> cross_terms;
    std::map<std::multiset<std::string>,std::vector<float>> bond_params;
    std::map<std::vector<std::string>,std::vector<float>> angle_params;
    std::map<std::vector<std::string>,std::pair<int,std::vector<float>>> dihedral_params;
    std::map<std::vector<std::string>,std::vector<float>> improper_params;

    std::vector<Vector3> forces;
    std::vector<float> energies;

    std::vector<bool> touched;

    Vector3 cellBasis1,cellBasis2,cellBasis3;


    float timestep;

    float cut_off = 4.0;

    float *atom_coords,
            *atom_masses_cpu,
            *velocities_cpu,
            *forces_cpu,
            *bond_param_cpu,
            *angle_param_cpu,
            *dihedral_param_cpu,
            *improper_param_cpu;
    float *atom_coords_gpu,
            *atom_masses_gpu,
            *velocities_gpu,
            *forces_gpu,
            *bond_param_gpu,
            *angle_param_gpu,
            *dihedral_param_gpu,
            *improper_param_gpu;

    int3 *cell_aligns;
    int *cell_lists;
    unsigned int *cell_list_size;
    int3 *max_cpu,*min_cpu;

    int3 *max_gpu,*min_gpu;

    int3 *cell_aligns_gpu;
    int *cell_lists_gpu;
    unsigned  int *cell_list_size_gpu;

    void BuildCellList();

    void AccumulateBondForce();
    void AccumulateAngleForce();
    void AccumulateDihedral();
    void AccumulateImproper();
    void AccumulateUnbonded();
    void AccumulateVelocityPosition();

    //GPU STEPS
    void initializeGPUMemoryFormat();
    void initializeGPUDevice();
    void singleGPUStep();




public:
    Atoms();
    void AddAtomFromString(std::string);
    void AddPSFInfoToAtom(std::string);
    void AddBond(int,int);
    void AddAngle(int,int,int);
    void AddDihedral(int,int,int,int);
    void AddImproper(int,int,int,int);
    void AddCrossTerm(int,int,int,int);

    void AddBondParams(std::string,std::string,float,float);
    void AddAngleParams(std::string,std::string,std::string,float,float);
    void AddAngleParams(std::string,std::string,std::string,float,float,float,float);
    void AddDihedralParams(std::string,std::string,std::string,std::string,
                        float,int,float);
    void AddImproperParams(std::string,std::string,std::string,std::string,
                        float,float);
    void RunSingleStep(float);
    void PrepareGPU(float);
    void RunSingleStepGPU();

    //validation
    void validation();

};


#endif //READ_PDB_ATOMS_H
