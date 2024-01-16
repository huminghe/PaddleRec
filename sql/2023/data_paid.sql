select a.distinct_id, a.author_id_str, a.country, a.author_country, a.time, c.groupId, a.manufacturer, a.height, a.model, a.duration_time, b.cnt from
(select distinct_id, author_id_str, duration_time, `time` as time, `$country` as country, author_country,
`$model` as model, `$manufacturer` as manufacturer, `$screen_height` as height from events where
event = 'task' and task_name = 'video_chat_call_end' and
((call_type in ('payZego', 'payVideo') and duration_time > 100) or (call_type in ('blurVideo', 'freeVideo') and duration_time > 15))
and author_id_str is not null
and `date` >='2023-03-01' and `date` < '2023-05-01') a
join
(select distinct_id, count(*) as cnt from events where
event = 'task' and task_name = 'purchase_pop'
and task_result = 'success'
and `date` >= '2022-01-01'
group by distinct_id) b
on a.distinct_id = b.distinct_id
left outer join
(select first_id, groupId from users) c
on a.distinct_id = c.first_id