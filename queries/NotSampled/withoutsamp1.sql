DROP MATERIALIZED VIEW IF EXISTS mimiciii.withoutsampl1;
CREATE MATERIALIZED VIEW mimiciii.withoutsampl1 AS

with lab1 as
(
	select *from mimiciii.overalltable_lab
),

dem1 as
(
 select *from mimiciii.demographics1 dem1
)

select *from lab1 where icustay_id in (select icustay_id from dem1)