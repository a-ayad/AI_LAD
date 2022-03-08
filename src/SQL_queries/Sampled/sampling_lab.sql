-- code is inspired from https://stackoverflow.com/questions/15576794/best-way-to-count-records-by-arbitrary-time-intervals-in-railspostgres/15577413#15577413
-- and https://stackoverflow.com/questions/27351079/query-aggregated-data-with-a-given-sampling-time

--This code samples the data within table 'overalltable' with a resolution of 4 hours. 

DROP MATERIALIZED VIEW IF EXISTS mimiciii.sampled_lab;
CREATE MATERIALIZED VIEW mimiciii.sampled_lab as

-- This part is in order to generate time binning.
WITH minmax as 
(
	SELECT subject_id, hadm_id, icustay_id , min(charttime) as mint, max(charttime) as maxt
	FROM mimiciii.overalltable_lab
	GROUP BY icustay_id, subject_id, hadm_id
	ORDER BY icustay_id, subject_id, hadm_id
), 
	
grid as 
(
	SELECT icustay_id, subject_id, hadm_id, generate_series(mint,maxt,interval '4 hours') as start_time
	FROM minmax
	GROUP BY icustay_id, subject_id, hadm_id,mint,maxt
    ORDER BY icustay_id, subject_id, hadm_id
)
	
SELECT ol.icustay_id, ol.subject_id, ol.hadm_id, start_time
	 --lab values
     ,AVG(HEMATOCRIT)AS HEMATOCRIT
	 ,AVG(PLATELET_COUNT)AS PLATELET_COUNT
	 ,AVG(HEMOGLOBIN)AS HEMOGLOBIN
	 ,AVG(MCHC)AS MCHC
	 ,AVG(RBC)AS RBC
	 ,AVG(MCH)AS MCH
	 ,AVG(MCV)AS MCV
	 ,AVG(RDW)AS RDW
	 ,AVG(PTT)AS PTT
	 ,AVG(INR)AS INR
	 ,AVG(PT)AS PT
	 ,AVG(LYMPHOCYTES)AS LYMPHOCYTES
	 ,AVG(NEUTROPHILS)AS NEUTROPHILS
	 ,AVG(MONOCYTES) AS MONOCYTES
	 ,AVG(EOSINOPHILS)AS EOSINOPHILS
	 ,AVG(BASOPHILS)AS BASOPHILS
	 ,AVG(BANDS)AS BANDS
	 ,AVG(HYPOCHROMIA)AS HYPOCHROMIA
	 ,AVG(ANISOCYTOSIS)AS ANISOCYTOSIS
	 ,AVG(MACROCYTES)AS MACROCYTES
	 ,AVG(ATYPICAL_LYMPHOCYTES)AS ATYPICAL_LYMPHOCYTES
	 ,AVG(METAMYELOCYTES)AS METAMYELOCYTES
	 ,AVG(MYELOCYTES)AS MYELOCYTES
	 ,AVG(MICROCYTES)AS MICROCYTES
	 ,AVG(POIKILOCYTOSIS)AS POIKILOCYTOSIS
	 ,AVG(POLYCHROMASIA)AS POLYCHROMASIA
	 ,AVG(FIBRINOGEN)AS FIBRINOGEN
	 ,AVG(PLATELET_SMEAR)AS PLATELET_SMEAR
	 ,AVG(GRANULOCYTE_COUNT)AS GRANULOCYTE_COUNT
	 ,AVG(NUCLEATED_RED_CELLS)AS NUCLEATED_RED_CELLS
	 ,AVG(OVALOCYTES)AS OVALOCYTES
	 ,AVG(UREA_NITROGEN)AS UREA_NITROGEN
	 ,AVG(SCHISTOCYTES)AS SCHISTOCYTES
	 ,AVG(RETICULOCYTE_COUNT)AS RETICULOCYTE_COUNT
	 ,AVG(TEARDROP_CELLS)AS TEARDROP_CELLS
	 ,AVG(BURR_CELLS)AS BURR_CELLS
	 ,AVG(SEDIMENTATION_RATE)AS SEDIMENTATION_RATE
	 ,AVG(LARGE_PLATELETS)AS LARGE_PLATELETS
	 ,AVG(TARGET_CELLS)AS TARGET_CELLS
	 ,AVG(FIBRIN_DEGRADATION_PRODUCTS)AS FIBRIN_DEGRADATION_PRODUCTS
	 ,AVG(D_DIMER)AS D_DIMER
	 ,AVG(RED_BLOOD_CELL_FRAGMENTS)AS RED_BLOOD_CELL_FRAGMENTS
	 ,AVG(BASOPHILIC_STIPPLING)AS BASOPHILIC_STIPPLING
	 ,AVG(UHOLD)AS UHOLD
	 ,AVG(BLASTS)AS BLASTS
	 ,AVG(POLYS)AS POLYS
	 ,AVG(WBC)AS WBC

FROM grid g

LEFT JOIN mimiciii.overalltable_lab  ol ON ol.charttime >= g.start_time
				   AND ol.charttime <  g.start_time + '4 hours'
				   AND ol.icustay_id=g.icustay_id
				   AND ol.subject_id=g.subject_id
				   AND ol.hadm_id=g.hadm_id

GROUP  BY ol.icustay_id,ol.subject_id, ol.hadm_id, start_time
ORDER  BY icustay_id,subject_id, hadm_id, start_time
