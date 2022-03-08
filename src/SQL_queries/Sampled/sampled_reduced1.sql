--- RETURN ONLY DATA WHERE IS NO READMITION

DROP MATERIALIZED VIEW IF EXISTS mimiciii.sampled_reduced1;
CREATE MATERIALIZED VIEW mimiciii.sampled_reduced1 AS

with lab1 as
(
	select *from mimiciii.sampled_lab
),

dem1 as
(
 select *from mimiciii.demographics1 dem1
)

select *from lab1 where icustay_id in (select icustay_id from dem1)