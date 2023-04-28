select a.distinct_id, a.author_id_str, a.country, a.author_country, a.time from
(select distinct_id, author_id_str, duration_time, `time` as time, `$country` as country, author_country from events where
event = 'task' and task_name = 'video_chat_call_end' and call_type in ('blurVideo', 'freeVideo')
and duration_time >= 10
and author_id_str is not null
and `date` >='2022-01-01') a
join
(select distinct_id, min(`time`) as first_time from events where
event = 'task' and task_name = 'purchase_pop'
and task_result = 'success' and (order_num = 1 or order_num = 0) and isFirstBuy = 1
and `date` >= '2022-01-01'
group by distinct_id) b
on a.distinct_id = b.distinct_id
where a.time < b.first_time



select distinct_id, author_id_str, `$country`, author_country, `time` from events where
event = 'task' and task_name = 'purchase_pop'
and task_result = 'success'
and (order_num = 1 or order_num = 0) and isFirstBuy = 1
and author_id_str is not null
and author_id_str != '' and author_id_str != '0' and author_id_str != 'null'
and `date` >= '2022-01-01'

