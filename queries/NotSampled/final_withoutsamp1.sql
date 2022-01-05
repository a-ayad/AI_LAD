--query script to merge the sampled data with the corresponding scores and demographic information

DROP MATERIALIZED VIEW IF EXISTS mimiciii.data1_v1;
CREATE MATERIALIZED VIEW mimiciii.data1_v1 AS

SELECT   --samp.* 

		fws.icustay_id, ic.subject_id , ic.hadm_id,  dem.first_admit_age,
		dem.gender, weig.weight, dem.icu_readm, dem.elixhauser_score 

		,fws.HEMATOCRIT,fws.PLATELET_COUNT,fws.HEMOGLOBIN,fws.MCHC,fws.RBC,fws.MCH,fws.MCV,fws.RDW,--fws.WBC, 
		fws.PTT, fws.INR, fws.PT, fws.LYMPHOCYTES,fws.NEUTROPHILS,--fws.MONOCYTES
		fws.EOSINOPHILS,fws.BASOPHILS,fws.BANDS,fws.HYPOCHROMIA,fws.ANISOCYTOSIS,fws.MACROCYTES
		,fws.ATYPICAL_LYMPHOCYTES,fws.METAMYELOCYTES,fws.MYELOCYTES,fws.MICROCYTES,fws.POIKILOCYTOSIS,fws.POLYCHROMASIA,fws.	FIBRINOGEN,fws.PLATELET_SMEAR,fws.GRANULOCYTE_COUNT,fws.NUCLEATED_RED_CELLS
		,fws.OVALOCYTES,fws.UREA_NITROGEN,fws.	SCHISTOCYTES,fws.RETICULOCYTE_COUNT,fws.TEARDROP_CELLS,fws.BURR_CELLS,fws.SEDIMENTATION_RATE,fws.LARGE_PLATELETS,fws.TARGET_CELLS
		,fws.FIBRIN_DEGRADATION_PRODUCTS,fws.D_DIMER,fws.RED_BLOOD_CELL_FRAGMENTS,fws.BASOPHILIC_STIPPLING,fws.UHOLD,fws.BLASTS,-- fws.WBC, fws.MONOCYTES, 
		fws.POLYS,dem.hospmort90day, dem.dischtime, dem.deathtime


FROM mimiciii.withoutsampl1 fws

LEFT JOIN mimiciii.demographics1 dem
ON fws.icustay_id=dem.icustay_id 

LEFT JOIN mimiciii.getweight2 weig
ON fws.icustay_id=weig.icustay_id

INNER JOIN mimiciii.icustays ic
ON fws.icustay_id=ic.icustay_id


ORDER BY fws.icustay_id, fws.subject_id, fws.hadm_id
--LIMIT 1000000

