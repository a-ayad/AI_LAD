-- This code is retrieved from https://github.com/MIT-LCP/mimic-code/blob/master/concepts/pivot/pivoted-lab.sql and modified according to lab list
DROP MATERIALIZED VIEW IF EXISTS mimiciii.getLabvalues1;
CREATE MATERIALIZED VIEW mimiciii.getLabvalues1 as
with le as 
(
  select le.subject_id , le.hadm_id, ic.icustay_id
    , le.charttime
	,(CASE WHEN itemid = 51249 and valuenum>0 and valuenum <100 THEN valuenum else null end) as MCHC 
	,(CASE WHEN itemid = 51279 and valuenum>0 and valuenum <100 THEN valuenum else null end) as RBC
	,(CASE WHEN itemid = 51248 and valuenum>0 and valuenum <100 THEN valuenum else null end) as MCH
	,(CASE WHEN itemid = 51250 and valuenum>0 and valuenum <100 THEN valuenum else null end) as MCV
	,(CASE WHEN itemid = 51277 and valuenum>0 and valuenum <100 THEN valuenum else null end) as RDW
	,(CASE WHEN itemid = 51244 and valuenum>0 and valuenum <100 THEN valuenum else null end) as LYMPHOCYTES
    ,(CASE WHEN itemid = 51256 and valuenum>0 and valuenum <100 THEN valuenum else null end) as NEUTROPHILS
	,(CASE WHEN itemid in (51120,51254) and valuenum>0 and valuenum <100 THEN valuenum else null end) as MONOCYTES
	,(CASE WHEN itemid = 51200 and valuenum>0 and valuenum <100 THEN valuenum else null end) as EOSINOPHILS
    ,(CASE WHEN itemid = 51146 and valuenum>0 and valuenum <100 THEN valuenum else null end) as BASOPHILS
	,(CASE WHEN itemid = 51144 and valuenum>0 and valuenum <100 THEN valuenum else null end) as BANDS
	,(CASE WHEN itemid = 51233 and valuenum>0 and valuenum <100 THEN valuenum else null end) as HYPOCHROMIA
	,(CASE WHEN itemid = 51137 and valuenum>0 and valuenum <100 THEN valuenum else null end) as ANISOCYTOSIS
	,(CASE WHEN itemid = 51246 and valuenum>0 and valuenum <100 THEN valuenum else null end) as MACROCYTES
	,(CASE WHEN itemid = 51143 and valuenum>0 and valuenum <100 THEN valuenum else null end) as ATYPICAL_LYMPHOCYTES
	,(CASE WHEN itemid = 51251 and valuenum>0 and valuenum <100 THEN valuenum else null end) as METAMYELOCYTES
	,(CASE WHEN itemid = 51255 and valuenum>0 and valuenum <100 THEN valuenum else null end) as MYELOCYTES
	,(CASE WHEN itemid = 51252 and valuenum>0 and valuenum <100 THEN valuenum else null end) as MICROCYTES
    ,(CASE WHEN itemid = 51267 and valuenum>0 and valuenum <100 THEN valuenum else null end) as POIKILOCYTOSIS
     ,(CASE WHEN itemid = 51268 and valuenum>0 and valuenum <100 THEN valuenum else null end) as POLYCHROMASIA
	 ,(CASE WHEN itemid = 51214 and valuenum>0 and valuenum <100 THEN valuenum else null end) as FIBRINOGEN
	 ,(CASE WHEN itemid = 51266 and valuenum>0 and valuenum <100 THEN valuenum else null end) as PLATELET_SMEAR
	 ,(CASE WHEN itemid = 51218 and valuenum>0 and valuenum <100 THEN valuenum else null end) as GRANULOCYTE_COUNT
	 ,(CASE WHEN itemid = 51257 and valuenum>0 and valuenum <100 THEN valuenum else null end) as NUCLEATED_RED_CELLS
	 ,(CASE WHEN itemid = 51260 and valuenum>0 and valuenum <100 THEN valuenum else null end) as OVALOCYTES
	 ,(CASE WHEN itemid = 51104 and valuenum>0 and valuenum <100 THEN valuenum else null end) as UREA_NITROGEN
	 ,(CASE WHEN itemid = 51287 and valuenum>0 and valuenum <100 THEN valuenum else null end) as SCHISTOCYTES
	 ,(CASE WHEN itemid = 51283 and valuenum>0 and valuenum <100 THEN valuenum else null end) as RETICULOCYTE_COUNT
	 ,(CASE WHEN itemid = 51296 and valuenum>0 and valuenum <100 THEN valuenum else null end) as TEARDROP_CELLS
     ,(CASE WHEN itemid = 51151 and valuenum>0 and valuenum <100 THEN valuenum else null end) as BURR_CELLS
	 ,(CASE WHEN itemid = 51288 and valuenum>0 and valuenum <100 THEN valuenum else null end) as SEDIMENTATION_RATE
     ,(CASE WHEN itemid = 51240  and valuenum>0 and valuenum <100 THEN valuenum else null end) as LARGE_PLATELETS
	,(CASE WHEN itemid = 51294 and valuenum>0 and valuenum <100 THEN valuenum else null end) as TARGET_CELLS
	,(CASE WHEN itemid = 51213  and valuenum>0 and valuenum <100 THEN valuenum else null end) as FIBRIN_DEGRADATION_PRODUCTS
	,(CASE WHEN itemid = 51196  and valuenum>0 and valuenum <100 THEN valuenum else null end) as D_DIMER
	,(CASE WHEN itemid = 51278 and valuenum>0 and valuenum <100 THEN valuenum else null end) as RED_BLOOD_CELL_FRAGMENTS
	,(CASE WHEN itemid = 51145  and valuenum>0 and valuenum <100 THEN valuenum else null end) as BASOPHILIC_STIPPLING
	,(CASE WHEN itemid = 51103  and valuenum>0 and valuenum <100 THEN valuenum else null end) as UHOLD
	,(CASE WHEN itemid = 51148 and valuenum>0 and valuenum <100 THEN valuenum else null end) as BLASTS
	,(CASE WHEN itemid = 51125 and valuenum>0 and valuenum <100 THEN valuenum else null end) as  POLYS
    , (CASE WHEN itemid in (50810,51221) and valuenum>0 and valuenum <100 THEN valuenum else null end) as HEMATOCRIT -- % 'HEMATOCRIT'
    , (CASE WHEN itemid in (50811,51222) and valuenum>0 and valuenum <50 THEN valuenum else null end) as HEMOGLOBIN -- g/dL 'HEMOGLOBIN'
    , (CASE WHEN itemid = 51275 and valuenum>0 and valuenum <150 THEN valuenum else null end) as PTT -- sec 'PTT'
    , (CASE WHEN itemid = 51237 and valuenum>0 and valuenum <50 THEN valuenum else null end) as INR -- 'INR'
    , (CASE WHEN itemid = 51274 and valuenum>0 and valuenum <150 THEN valuenum else null end) as PT -- sec 'PT'
    , (CASE WHEN itemid in (51300,51301,51128) and valuenum>0 and valuenum <1000 THEN valuenum else null end) as WBC -- 'WBC'
	, (CASE WHEN itemid in (51265) and valuenum>0 and valuenum <1000 THEN valuenum else null end) as PLATELET_COUNT -- 'PLATELET COUNT'



	
	
    --ELSE le.valuenum
  from mimiciii.labevents le
	-- LABEVENTS do not have a icustay_id recorded. However, that can be obtained using clues such as the subject_id and hadm_id; and comparing the charttime of the measurement with an icustay time.
	-- This idea of adding icustays has been retrieved from https://github.com/MIT-LCP/mimic-code/blob/master/concepts/firstday/blood-gas-first-day.sql.
    left join mimiciii.icustays ic
    on le.subject_id = ic.subject_id and le.hadm_id = ic.hadm_id
    and le.charttime between (ic.intime - interval '6' hour) and (ic.intime + interval '1' day)
   -- where ce.error IS DISTINCT FROM 1
  where le.itemid in
   -- comment is: LABEL | CATEGORY | FLUID | NUMBER OF ROWS IN LABEVENTS
(51221	,--	Hematocrit
51265	,	--Platelet Count
51222	,	--Hemoglobin
51249	,	--MCHC
51279	,	--Red Blood Cells
51248	,	--MCH
51250	,	--MCV
51277	,	--RDW
51301	,	--White Blood Cells
51275	,	--PTT
51237	,	--INR(PT)
51274	,	--PT
51244	,	--Lymphocytes
51256	,	--Neutrophils
51254	,	--Monocytes
51200	,	--Eosinophils
51146	,	--Basophils
51144	,	--Bands
51233	,	--Hypochromia
51137	,	--Anisocytosis
51246	,	--Macrocytes
51143	,	--Atypical Lymphocytes
51251	,	--Metamyelocytes
51255	,	--Myelocytes
51252	,	--Microcytes
51267	,	--Poikilocytosis
51268	,	--Polychromasia
51214	,	--Fibrinogen
51266	,	--Platelet Smear
51218	,	--Granulocyte
51257	,	--Nucleated Red Cells
51260	,	--Ovalocytes
51104	,	--Urea Nitrogen
51287	,	--Schistocytes
51283	,	--Reticulocyte
51296	,	--Teardrop Cells
51151	,	--Burr Cells
51288	,	--Sedimentation Rate
51240	,	--Large Platelets
51294	,	--Target Cells
51213	,	--Fibrin Degradation Products
51196	,	--D-Dimer
51278	,	--Red Blood Cell Fragments
51145	,	--Basophilic Stippling
51103	,	--Uhold
51148	,	--Blasts
51128	,	--WBC
51116	,	--Lymphocytes
51120	,	--Monocytes
51254	,	--Monocytes
51125		--Polys

	 ) 
  
)


  select
  	subject_id
  , hadm_id	
  , icustay_id
  , charttime
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
,AVG(WBC)AS WBC
,AVG(MONOCYTES)AS MONOCYTES
,AVG(POLYS)AS POLYS

  from le
  group by le.icustay_id,le.subject_id,le.hadm_id,  le.charttime ;
         


