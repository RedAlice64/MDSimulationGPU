//
// Created by xiuqi on 11/28/16.
//

#ifndef READ_PDB_ATOM_H
#define READ_PDB_ATOM_H

#include <string>
#include "Vector3.h"

class Atom{
private:
    int atom_id;
    std::string atom_name;
    std::string residue_name;
    int residue_id;
    float x,y,z;
    float occupancy;
    float beta;
    std::string segment_name;
    Vector3 coordinate;
    Vector3 velocity;

//psf data
    std::string atom_type;
    float charge;

public:
    float mass;

    Atom();
    Atom(std::string);
    void AddPSFData(float,float,std::string);


    float getX() const {
        return coordinate.getX();
    }

    float getY() const {
        return coordinate.getY();
    }

    float getZ() const {
        return coordinate.getZ();
    }

    Vector3 getXYZ() const {
        return coordinate;
    }

    const std::string &getAtom_type() const {
        return atom_type;
    }

    inline void intergrateCoordinate(float timestep){
        Vector3 delta(velocity,0.5f * timestep);
        coordinate += delta;
    }

    inline void intergrateVelocity(Vector3 &force,float timestep){
        Vector3 delta(force,timestep / mass);
        velocity += delta;
    }

};




#endif //READ_PDB_ATOM_H
