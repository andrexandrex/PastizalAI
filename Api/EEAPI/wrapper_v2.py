#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Version: v1.2
Date: 2021-04-01
Authors: Mullissa A., Vollrath A., Braun, C., Slagter B., Balling J., Gou Y., Gorelick N.,  Reiche J.
Description: A wrapper function to derive the Sentinel-1 ARD
"""


import ee
import border_noise_correction as bnc
import speckle_filter as sf
import terrain_flattening as trf
import helper
import re



###########################################
# DO THE JOB
###########################################
# Function to cast all bands of an image to Float32
def cast_to_float32(image):
    return image.toFloat()

# Apply this function to each image in the collection before exporting


def s1_preproc(params,index,idx,  system_index):
    """
    Applies preprocessing to a collection of S1 images to return an analysis ready sentinel-1 data.

    Parameters
    ----------
    params : Dictionary
        These parameters determine the data selection and image processing parameters.

    Raises
    ------
    ValueError
        

    Returns
    -------
    ee.ImageCollection
        A processed Sentinel-1 image collection

    """


    APPLY_BORDER_NOISE_CORRECTION = params['APPLY_BORDER_NOISE_CORRECTION']
    APPLY_TERRAIN_FLATTENING = params['APPLY_TERRAIN_FLATTENING']
    APPLY_SPECKLE_FILTERING = params['APPLY_SPECKLE_FILTERING']
    POLARIZATION = params['POLARIZATION']
    SPECKLE_FILTER_FRAMEWORK = params['SPECKLE_FILTER_FRAMEWORK']
    SPECKLE_FILTER = params['SPECKLE_FILTER']
    SPECKLE_FILTER_KERNEL_SIZE = params['SPECKLE_FILTER_KERNEL_SIZE']
    SPECKLE_FILTER_NR_OF_IMAGES = params['SPECKLE_FILTER_NR_OF_IMAGES']
    TERRAIN_FLATTENING_MODEL = params['TERRAIN_FLATTENING_MODEL']
    DEM = params['DEM']
    TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER = params['TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER']
    FORMAT = params['FORMAT']
    START_DATE = params['START_DATE']
    STOP_DATE = params['STOP_DATE']
    ROI = params['ROI']
    CLIP_TO_ROI = params['CLIP_TO_ROI']
    SAVE_ASSET = params['SAVE_ASSET']
    ###########################################
    # 0. CHECK PARAMETERS
    ###########################################

    if APPLY_BORDER_NOISE_CORRECTION is None:
        APPLY_BORDER_NOISE_CORRECTION = True
    if APPLY_TERRAIN_FLATTENING is None:
        APPLY_TERRAIN_FLATTENING = True
    if APPLY_SPECKLE_FILTERING is None:
        APPLY_SPECKLE_FILTERING = True
    if POLARIZATION is None:
        POLARIZATION = 'VVVH'
    if SPECKLE_FILTER_FRAMEWORK is None:
        SPECKLE_FILTER_FRAMEWORK = 'MULTI BOXCAR'
    if SPECKLE_FILTER is None:
        SPECKLE_FILTER = 'GAMMA MAP'
    if SPECKLE_FILTER_KERNEL_SIZE is None:
        SPECKLE_FILTER_KERNEL_SIZE = 7
    if SPECKLE_FILTER_NR_OF_IMAGES is None:
        SPECKLE_FILTER_NR_OF_IMAGES = 10
    if TERRAIN_FLATTENING_MODEL is None:
        TERRAIN_FLATTENING_MODEL = 'VOLUME'
    if TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER is None:
        TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER = 0
    if FORMAT is None:
        FORMAT = 'DB'

    pol_required = ['VV', 'VH', 'VVVH']
    if (POLARIZATION not in pol_required):
        raise ValueError("ERROR!!! Parameter POLARIZATION not correctly defined")


    model_required = ['DIRECT', 'VOLUME']
    if (TERRAIN_FLATTENING_MODEL not in model_required):
        raise ValueError("ERROR!!! Parameter TERRAIN_FLATTENING_MODEL not correctly defined")

    format_required = ['LINEAR', 'DB']
    if (FORMAT not in format_required):
        raise ValueError("ERROR!!! FORMAT not correctly defined")

    frame_needed = ['MONO', 'MULTI']
    if (SPECKLE_FILTER_FRAMEWORK not in frame_needed):
        raise ValueError("ERROR!!! SPECKLE_FILTER_FRAMEWORK not correctly defined")

    format_sfilter = ['BOXCAR', 'LEE', 'GAMMA MAP'
              ,'REFINED LEE', 'LEE SIGMA']
    if (SPECKLE_FILTER not in format_sfilter):
        raise ValueError("ERROR!!! SPECKLE_FILTER not correctly defined")

    if (TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER < 0):
        raise ValueError("ERROR!!! TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER not correctly defined")

    if (SPECKLE_FILTER_KERNEL_SIZE <= 0):
        raise ValueError("ERROR!!! SPECKLE_FILTER_KERNEL_SIZE not correctly defined")
    ###########################################
    # 1. DATA SELECTION
    ###########################################

    # select S-1 image collection
    '''
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD')\
        .filterDate(START_DATE, STOP_DATE) \
        .filterBounds(ROI)'''
   
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD/{system_index}') \
        .filterBounds(ROI)
    # select polarization
    '''
    if (POLARIZATION == 'VV'):
        s1 = s1.select(['VV', 'angle'])
    elif (POLARIZATION == 'VH'):
        s1 = s1.select(['VH', 'angle'])
    elif (POLARIZATION == 'VVVH'):
        s1 = s1.select(['VV', 'VH', 'angle'])
        
    print('Number of images in collection: ', s1.size().getInfo())'''
    def safe_select(image, pol):
      band_names = image.bandNames()
      def maybe_select_band(band):
          return ee.Algorithms.If(band_names.contains(band), [band], [])
      
      bands_to_select = ee.List(maybe_select_band('VV')) \
          .cat(ee.List(maybe_select_band('VH'))) \
          .cat(ee.List(maybe_select_band('angle')))
      
      return image.select(bands_to_select)

    
    # Apply polarization filtering
    if POLARIZATION == 'VV':
        s1 = s1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    elif POLARIZATION == 'VH':
        s1 = s1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    elif POLARIZATION == 'VVVH':
        s1 = s1.filter(ee.Filter.And(
            ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'),
            ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')
        ))

    # Check for empty collection before mapping
    if s1.size().getInfo() == 0:
        print(f"No images found for index {index}. Skipping.")
        return
    else:
        print(f"Number of images in collection: {s1.size().getInfo()}")

    # Apply safe band selection AFTER checking size
    s1 = s1.map(lambda img: safe_select(img, POLARIZATION))


    ###########################################
    # 2. ADDITIONAL BORDER NOISE CORRECTION
    ###########################################

    if (APPLY_BORDER_NOISE_CORRECTION):
        s1_1 = s1.map(bnc.f_mask_edges)
        print('Additional border noise correction is completed')
    else:
        s1_1 = s1
    ########################
    # 3. SPECKLE FILTERING
    #######################



    if (APPLY_SPECKLE_FILTERING):
        if (SPECKLE_FILTER_FRAMEWORK == 'MONO'):
            s1_1 = ee.ImageCollection(sf.MonoTemporal_Filter(s1_1, SPECKLE_FILTER_KERNEL_SIZE, SPECKLE_FILTER))
            s1_1 = s1_1.map(lambda img: img.toFloat())
            print('Mono-temporal speckle filtering is completed')
        else:
            s1_1 = ee.ImageCollection(sf.MultiTemporal_Filter(s1_1, SPECKLE_FILTER_KERNEL_SIZE, SPECKLE_FILTER, SPECKLE_FILTER_NR_OF_IMAGES))
            
            print('Multi-temporal speckle filtering is completed')

    ########################
    # 4. TERRAIN CORRECTION
    #######################

    if (APPLY_TERRAIN_FLATTENING):
        s1_1 = (trf.slope_correction(s1_1 
                                    ,TERRAIN_FLATTENING_MODEL
                                        ,DEM
                                                ,TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER))
        print('Radiometric terrain normalization is completed')

    ########################
    # 5. OUTPUT
    #######################

    if (FORMAT == 'DB'):
        s1_1 = s1_1.map(helper.lin_to_db)
        
        
    #clip to roi
    if (CLIP_TO_ROI):
        s1_1 = s1_1.map(lambda image: image.clip(ROI))
         
    s1_1 = s1_1.map(cast_to_float32)    


    if (SAVE_ASSET): 
            
        size = s1_1.size().getInfo()#quantity of images
        imlist = s1_1.toList(size)

        folder = 'oefa_img_v5_tiff'
        for idx in range(size):
            img = ee.Image(imlist.get(idx))
            
            # Instead of relying on img.id(), just generate a date-based name safely
            date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            # Figure out which bands remain
            final_bands_list = img.bandNames().getInfo()  # e.g. ["VV","VH","angle"]
            # We only want to reflect the polarizations in the filename
            # because 'angle' is just the incidence angle band.
            # We'll build something like "VVVH", "VV", or "VH"
            # depending on what's actually there.
            has_vv = 'VV' in final_bands_list
            has_vh = 'VH' in final_bands_list
            if has_vv and has_vh:
                pol_label = 'VVVH'
            elif has_vv:
                pol_label = 'VV'
            elif has_vh:
                pol_label = 'VH'
            else:
                pol_label = 'NONE'  # Should not happen, but just in case

            
            export_name = f"{index}_{idx}_{date}_S1_GRD_{pol_label}"

            export_image = img.clip(ROI)

            task = ee.batch.Export.image.toDrive(
                image=export_image,
                description=export_name,
                folder=folder,
                region=ROI,  # Use the same geometry you clipped wit
                fileFormat='GeoTIFF',
                maxPixels=1e13
            )
            task.start()
            print(f'Exporting {export_name} to {folder}')
        # Return the final processed collection
    return s1_1
