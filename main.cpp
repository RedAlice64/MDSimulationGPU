#include <iostream>
#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include "atoms.h"

using std::cin;
using std::cout;
using std::getline;
using std::string;
using std::stringstream;
using std::vector;
int main() {

    //freopen("result", "w", stdout);

    Atoms atoms;


    std::ifstream pdb_file("apoa1.pdb");
    std::ifstream psf_file("apoa1.psf");
    std::ifstream param_file("par_all22_prot_lipid.xplor");

    for(string line;getline(pdb_file, line);){
        stringstream ss;
        ss<<line;
        string header;
        ss>>header;
        if(header.compare("ATOM"))continue;
        atoms.AddAtomFromString(line);

    }


    for(string line;getline(param_file,line);){
        stringstream ss;
        ss << line;
        string word;
        ss>>word;
        if(word.compare("BOND") == 0){
            string a,b;
            float Kb,b0;
            ss>>a>>b>>Kb>>b0;
            atoms.AddBondParams(a,b,Kb,b0);
        }
        else if(word.compare("ANGLE") == 0){

            string a,b,c,is_ub;
            float KTheta,Theta0,Kub,S0;
            ss >> a >> b >> c >> KTheta >> Theta0 >> is_ub;
            if(is_ub[0] != '!'){
                ss>>Kub>>S0;
                atoms.AddAngleParams(a,b,c,KTheta,Theta0,Kub,S0);
            }
            else atoms.AddAngleParams(a,b,c,KTheta,Theta0);
        }
        else if(word.compare("DIHEDRAL") == 0){
            string a,b,c,d,lines[4];
            float Kchi,delta;
            int n;
            ss>>a>>b>>c>>d>>lines[0]>>lines[1]>>lines[2]>>lines[3];
            stringstream local_ss;

            if(lines[0].size() > 5){
                local_ss<<lines[1]<<" "<<lines[2]<<" "<<lines[3];
                local_ss>>Kchi>>n>>delta;
            }
            else{
                local_ss<<lines[0]<<" "<<lines[1]<<" "<<lines[2];
                local_ss>>Kchi>>n>>delta;
            }
            atoms.AddDihedralParams(a,b,c,d,Kchi,n,delta);
        }
        else if(word.compare("IMPROPER") == 0){
            string a,b,c,d,blank;
            float Kpsi, psi0;
            ss >> a >> b >> c >> d >>Kpsi >> blank >> psi0;
            atoms.AddImproperParams(a, b,c,d, Kpsi, psi0);
        }
        //cout<<line<<std::endl;
    }

    for(string line;getline(psf_file,line);){
        if(line.size() < 2)continue;

        stringstream ss;
        ss<<line;
        string part1,part2;
        ss>>part1>>part2;
        if(part2.compare("!NATOM") == 0){
            while(getline(psf_file,line) && line.size() > 0){
                atoms.AddPSFInfoToAtom(line);
            }
        }
        else if(part2.compare("!NBOND:") == 0){
            while(getline(psf_file,line) && line.size() > 0){
                stringstream inner_ss;
                inner_ss<<line;
                for(int i = 0;i < 4;i++){
                    int a,b;
                    inner_ss>>a>>b;
                    atoms.AddBond(a,b);
                }
            }
        }
        else if(part2.compare("!NTHETA:") == 0){
            while(getline(psf_file,line) && line.size() > 0){
                stringstream inner_ss;
                inner_ss<<line;
                for(int i = 0;i < 3;i++){
                    int a,b,c;
                    inner_ss>>a>>b>>c;
                    atoms.AddAngle(a,b,c);
                }
            }
        }
        else if(part2.compare("!NPHI:") == 0){
            while(getline(psf_file,line) && line.size() > 0){
                stringstream inner_ss;
                inner_ss<<line;
                for(int i = 0;i < 2;i++){
                    int a,b,c,d;
                    inner_ss>>a>>b>>c>>d;
                    atoms.AddDihedral(a,b,c,d);
                }
            }
        }
        else if(part2.compare("!NIMPHI:") == 0){
            while(getline(psf_file,line) && line.size() > 0){
                stringstream inner_ss;
                inner_ss<<line;
                for(int i = 0;i < 2;i++){
                    int a,b,c,d;
                    inner_ss>>a>>b>>c>>d;
                    atoms.AddImproper(a,b,c,d);
                }
            }
        }
        else if(part2.compare("!NCRTERM:") == 0){
            while(getline(psf_file,line) && line.size() > 0){
                stringstream inner_ss;
                inner_ss<<line;
                for(int i = 0;i < 2;i++){
                    int a,b,c,d;
                    inner_ss>>a>>b>>c>>d;
                    atoms.AddCrossTerm(a,b,c,d);
                }
            }
        }
        //cout<<line<<std::endl;
    }
    atoms.PrepareGPU(0.000001);
    for(int i = 0;i <0;i++){
        atoms.RunSingleStep(1);
        cout<<"step "<<i<<" finished"<<std::endl;
    }

    for(int i = 0;i < 2;i++){
        atoms.RunSingleStepGPU();
        cout<<"gpu step "<<i<<" finished"<<std::endl;
    }
    atoms.validation();

    return 0;
}