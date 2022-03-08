DROP MATERIALIZED VIEW IF EXISTS mimiciii.withoutsampl2;
CREATE MATERIALIZED VIEW mimiciii.withoutsampl2 AS

with lab1 as
(
	select *from mimiciii.overalltable_lab
),

dem1 as
(
 select *from mimiciii.icustay_detail ic
 where icustay_seq=1
)

select *from lab1 where icustay_id in (select icustay_id from dem1)