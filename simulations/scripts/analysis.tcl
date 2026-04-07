#!/usr/bin/env tclsh

proc getFrameRange {filename} {
    set fh [open $filename "r"]
    set line [gets $fh]
    close $fh
    set frames [split $line " "]
    return [list [expr {int([lindex $frames 0])}] [expr {int([lindex $frames 1])}]]
}

proc loadMolecule {psfFile pdbFile dcdFile firstFrame lastFrame} {
    mol new $psfFile
    mol addfile $pdbFile
    mol addfile $dcdFile type dcd first $firstFrame last $lastFrame step 1 waitfor all
}

proc writeXYZ {selExpr outputFile} {
    set sel [atomselect top $selExpr]
    animate write xyz $outputFile sel $sel
    $sel delete
}

proc computeRDF {numFrames delta rmax outputFile} {
    set numBins [expr {int($rmax / $delta)}]
    set rdfData {}
    for {set i 0} {$i < $numBins} {incr i} {
        lappend rdfData [list 0 0 0]
    }
    for {set frame 1} {$frame <= $numFrames} {incr frame} {
        set selPolymer [atomselect top "type EO" frame $frame]
        set selWater [atomselect top "type P4" frame $frame]
        set rdf [measure gofr $selPolymer $selWater delta $delta rmax $rmax usepbc 1]
        set r_values    [lindex $rdf 0]
        set g_values    [lindex $rdf 1]
        set coor_values [lindex $rdf 2]
        set len [llength $r_values]
        for {set i 0} {$i < $len} {incr i} {
            set r [lindex $r_values $i]
            set g [lindex $g_values $i]
            set coor [lindex $coor_values $i]
            set bin [expr int($r / $delta)]
            if {$bin >= $numBins} { set bin [expr {$numBins - 1}] }
            set binData [lindex $rdfData $bin]
            set sumG    [expr [lindex $binData 0] + $g]
            set sumCoord [expr [lindex $binData 1] + $coor]
            set count   [expr [lindex $binData 2] + 1]
            set rdfData [lreplace $rdfData $bin $bin [list $sumG $sumCoord $count]]
        }
        $selPolymer delete
        $selWater delete
    }
    set rdfout [open $outputFile w]
    for {set i 0} {$i < $numBins} {incr i} {
        set binData [lindex $rdfData $i]
        set r_val [expr $delta * $i]
        set avgG 0
        set avgCoord 0
        if {[lindex $binData 2] > 0} {
            set avgG [expr [lindex $binData 0] / double([lindex $binData 2])]
            set avgCoord [expr [lindex $binData 1] / double([lindex $binData 2])]
        }
        puts $rdfout [format "%.1f %s %s" $r_val $avgG $avgCoord]
    }
    close $rdfout
}

proc calc_mean {values} {
    set total 0
    foreach val $values {
        set total [expr {$total + $val}]
    }
    return [expr {$total / double([llength $values])}]
}

proc calc_stddev {values mean} {
    set sumSq 0
    foreach val $values {
        set diff [expr {$val - $mean}]
        set sumSq [expr {$sumSq + $diff * $diff}]
    }
    return [expr {sqrt($sumSq / double([llength $values]))}]
}

proc computeRg {numFrames numResidues outputFile} {
    set avg_rg_values {}
    set std_rg_values {}
    for {set frame 1} {$frame <= $numFrames} {incr frame} {
        set frame_rg_values {}
        for {set j 1} {$j <= $numResidues} {incr j} {
            set sel [atomselect top "resname LIG and resid $j" frame $frame]
            lappend frame_rg_values [measure rgyr $sel weight mass]
            $sel delete
        }
        set avg_rg [calc_mean $frame_rg_values]
        set std_rg [calc_stddev $frame_rg_values $avg_rg]
        lappend avg_rg_values $avg_rg
        lappend std_rg_values $std_rg
    }
    set outfile [open $outputFile w]
    foreach avg_rg $avg_rg_values std_rg $std_rg_values {
        puts $outfile "$avg_rg $std_rg"
    }
    set overall_avg_rg [calc_mean $avg_rg_values]
    set overall_std_rg [calc_stddev $avg_rg_values $overall_avg_rg]
    puts $outfile "$overall_avg_rg $overall_std_rg"
    close $outfile
}

set frameRange [getFrameRange "frames.txt"]
lassign $frameRange begFrame endFrame

loadMolecule "20_interfaceafterpgn.psf" "20_interfaceafterpgn.pdb" "system.dcd" $begFrame $endFrame

writeXYZ "not name W WAF" "xyz.xyz"

set delta 0.1
set rmax 20.0
set totalFrames 70
computeRDF $totalFrames $delta $rmax "rdf.txt"

package require pbctools

set numResidues 20
computeRg $totalFrames $numResidues "rg.txt"

exit

