#!/usr/bin/env bash

echo fabsim $1 single_run_vvuq:$PWD,solver=$4,processes=$5,nodes=$6,cpus_per_process=$7,wall_time=$8,neso_outfile=$9

pwd

cp $3 .

fabsim $1 single_run_vvuq:$PWD,solver=$4,processes=$5,nodes=$6,cpus_per_process=$7,wall_time=$8,neso_outfile=$9
