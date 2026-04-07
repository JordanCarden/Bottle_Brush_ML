mol new interfaceafterpgn.psf
mol addfile interfaceafterpgn.pdb waitfor all

set trajectory_file "system.dcd"
mol addfile $trajectory_file type dcd waitfor all

set num_frames [molinfo top get numframes]
if {$num_frames == 0} {
    puts "Error: No frames found in system.dcd"
    exit
}

set lig_atoms [atomselect top "resname LIG"]
if {![llength [$lig_atoms list]]} {
    puts "Error: No atoms found with resname LIG"
    exit
}

$lig_atoms set chain A

mol representation VDW
mol selection "resname LIG"
mol addrep top

animate goto [expr {$num_frames - 1}]
set last_frame [expr {$num_frames - 1}]

$lig_atoms writepdb single_polymer.pdb

exit

