Install FreeSurfer from https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
export FREESURFER_HOME=/Applications/freesurfer/7.3.2
source ~ % source $FREESURFER_HOME/SetUpFreeSurfer.sh

#!/bin/bash

input_dir="INPUT_DIR"
output_dir="OUTPUT_DIR"

for i in {0..580}; do

    # Convert to FreeSurfer format
    mri_convert "${input_dir}/image_${i}_skull_removed.nii" "${output_dir}/subject${i}/mri/image_skull_removed.mgz"

    # Compute transform
    mri_robust_register --mov ${output_dir}/subject${i}/mri/image_skull_removed.mgz --dst ${output_dir}/subject${i}/fsaverage/mri/brain.mgz --lta ${output_dir}/subject${i}/mri/transform.lta --satit --vox2vox --cost mi

    # Apply transform
    mri_vol2vol --mov ${output_dir}/subject${i}/mri/image_skull_removed.mgz --targ ${output_dir}/subject${i}/fsaverage/mri/brain.mgz --o ${output_dir}/subject${i}/mri/image_transformed.mgz --lta ${output_dir}/subject${i}/mri/transform.lta --interp trilin

    # Intensity normalization
    mri_normalize  ${output_dir}/subject${i}/mri/image_transformed.mgz ${output_dir}/subject${i}/mri/image_norm.mgz -rmin 0 -rmax 1

    # Apply Transformation to the mask
    mri_convert ${input_dir}/seg_${i}.nii ${output_dir}/subject${i}/mri/seg.mgz

    mri_vol2vol --mov ${output_dir}/subject${i}/mri/seg.mgz --targ ${output_dir}/subject${i}/fsaverage/mri/brain.mgz --o ${output_dir}/subject${i}/mri/seg_transformed.mgz --lta ${output_dir}/subject${i}/mri/transform.lta --interp nearest

done
