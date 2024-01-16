select a.distinct_id, a.author_id_str, a.country, a.author_country, a.time, c.groupId, a.manufacturer, a.height, a.model, a.duration_time from
(select distinct_id, author_id_str, duration_time, `time` as time, `$country` as country, author_country,
`$model` as model, `$manufacturer` as manufacturer, `$screen_height` as height from events where
event = 'click' and click_name = 'video_call' and p_product != 'manager' and coins_num >=80
and author_id_str is not null
and `date` >='2023-06-01' and `date` < '2023-08-01') a
left outer join
(select first_id, groupId from users) c
on a.distinct_id = c.first_id
