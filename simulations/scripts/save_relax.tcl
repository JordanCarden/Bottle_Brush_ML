mol new relaxafterpgn.psf
mol addfile relaxafterpgn.pdb waitfor all

set trajectory_file "system.dcd"
mol addfile $trajectory_file type dcd waitfor all

set num_frames [molinfo top get numframes]
if {$num_frames == 0} {
    puts "Error: No frames found in system.dcd"
    exit
}

set all_atoms [atomselect top "all"]
if {![llength [$all_atoms list]]} {
    puts "Error: Could not select any atoms"
    exit
}

$all_atoms set chain X

animate goto [expr {$num_frames - 1}]
set last_frame [expr {$num_frames - 1}]

animate write pdb relax.pdb beg $last_frame end $last_frame sel $all_atoms

exit

