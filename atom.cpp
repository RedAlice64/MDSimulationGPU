//
// Created by xiuqi on 11/28/16.
//

#include "atom.h"

#include <iostream>
#include <sstream>

Atom::Atom() { }

Atom::Atom(std::string line) {
    std::stringstream ss;
    std::string temp;
    ss<<line;
    ss>>temp>>atom_id>>atom_name>>residue_name>>residue_id>>x>>y>>z>>occupancy>>beta>>segment_name;
    coordinate = Vector3(x,y,z);
    velocity = Vector3(0.0,0.0,0.0);
}

void Atom::AddPSFData(float m, float c, std::string type) {
    mass = m;
    charge = c;
    atom_type = type;
}