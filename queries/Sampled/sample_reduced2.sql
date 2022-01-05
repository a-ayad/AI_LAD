--- RETURN ONLY DATA WITH FIRST ICU_STAY 

DROP MATERIALIZED VIEW IF EXISTS mimiciii.sampled_reduced2;
CREATE MATERIALIZED VIEW mimiciii.sampled_reduced2 AS

with lab1 as
(
	select *from mimiciii.sampled_lab
),

dem1 as
(
 select *from mimiciii.icustay_detail dem1
  where icustay_seq=1
)

select *from lab1 where icustay_id in (select icustay_id from dem1)