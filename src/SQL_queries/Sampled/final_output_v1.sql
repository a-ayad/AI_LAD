--query script to merge the sampled data with the corresponding scores and demographic information

DROP MATERIALIZED VIEW IF EXISTS mimiciii.sampled_data1;
CREATE MATERIALIZED VIEW mimiciii.sampled_data1 AS

SELECT   --samp.* 

		samp.icustay_id, ic.subject_id , ic.hadm_id, samp.start_time , dem.first_admit_age,
		dem.gender, weig.weight, dem.icu_readm, dem.elixhauser_score 

		,samp.HEMATOCRIT,samp.PLATELET_COUNT,samp.HEMOGLOBIN,samp.MCHC,samp.RBC,samp.MCH,samp.MCV,samp.RDW,--samp.WBC, 
		samp.PTT, samp.INR, samp.PT, samp.LYMPHOCYTES,samp.NEUTROPHILS,--samp.MONOCYTES
		samp.EOSINOPHILS,samp.BASOPHILS,samp.BANDS,samp.HYPOCHROMIA,samp.ANISOCYTOSIS,samp.MACROCYTES
		,samp.ATYPICAL_LYMPHOCYTES,samp.METAMYELOCYTES,samp.MYELOCYTES,samp.MICROCYTES,samp.POIKILOCYTOSIS,samp.POLYCHROMASIA,samp.	FIBRINOGEN,samp.PLATELET_SMEAR,samp.GRANULOCYTE_COUNT,samp.NUCLEATED_RED_CELLS
		,samp.OVALOCYTES,samp.UREA_NITROGEN,samp.	SCHISTOCYTES,samp.RETICULOCYTE_COUNT,samp.TEARDROP_CELLS,samp.BURR_CELLS,samp.SEDIMENTATION_RATE,samp.LARGE_PLATELETS,samp.TARGET_CELLS
		,samp.FIBRIN_DEGRADATION_PRODUCTS,samp.D_DIMER,samp.RED_BLOOD_CELL_FRAGMENTS,samp.BASOPHILIC_STIPPLING,samp.UHOLD,samp.BLASTS,-- samp.WBC, samp.MONOCYTES, 
		samp.POLYS,dem.hospmort90day, dem.dischtime, dem.deathtime

FROM mimiciii.sampled_reduced1 samp
--FROM mimiciii.sampled_reduced samp


LEFT JOIN mimiciii.demographics1 dem
ON samp.icustay_id=dem.icustay_id 

LEFT JOIN mimiciii.getweight2 weig
ON samp.icustay_id=weig.icustay_id

INNER JOIN mimiciii.icustays ic
ON samp.icustay_id=ic.icustay_id


ORDER BY samp.icustay_id, samp.subject_id, samp.hadm_id, samp.start_time
--LIMIT 1000000

