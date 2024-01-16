
select a.distinct_id, a.author_id_str, a.country, a.author_country, a.time,
c.campaignId, c.groupId, a.manufacturer, a.height, a.model, a.p_product, a.duration_time, b.cnt from
(select distinct_id, author_id_str, duration_time, `time` as time, `$country` as country, author_country,
`$model` as model, `$manufacturer` as manufacturer, `$screen_height` as height, library, p_product from events where
event = 'task' and task_name in ('video_chat_call_end', 'video_matchX_call_end') and
((library = 'payZego' and duration_time > 110) or (library = 'blurVideo' and duration_time > 14)
or (library = 'freeVideo' and duration_time > 28) or (library = 'freeZego' and duration_time > 28) or (library = 'payVideo' and duration_time > 56))
and author_id_str is not null
and `date` >='2023-04-01' and `date` < '2023-09-01') a
join
(select distinct_id, count(*) as cnt from events where
event = 'task' and task_name = 'purchase_pop'
and task_result = 'success'
and `date` >= '2022-01-01'
group by distinct_id) b
on a.distinct_id = b.distinct_id
left outer join
(select first_id, groupId, campaignId from users) c
on a.distinct_id = c.first_id
