
select a.distinct_id, a.country, a.author_id_str, a.author_country, a.cnt from
(select distinct_id, author_id_str, max(`time`) as time, max(`$country`) as country, max(author_country) as author_country,
max(`$model`) as model, max(`$manufacturer`) as manufacturer, max(`$screen_height`) as height, count(*) as cnt from events where
event = 'card_show' and page_url = 'meet' and author_id_str is not null and pay_user = 1
and author_id_str is not null and author_id_str != ''
and `date` >'2023-07-04'
and (distinct_id like '0%' or distinct_id like '1%')
group by distinct_id, author_id_str) a
left outer join
(select distinct_id, author_id_str, count(*) as cnt from events where
event = 'click' and click_element = 'CARD' and page_url = 'meet' and pay_user = 1
and `date` >= '2023-06-04'
group by distinct_id, author_id_str) b
on a.distinct_id = b.distinct_id and a.author_id_str = b.author_id_str
left outer join
(select distinct_id, author_id_str, count(*) as cnt from events where
event = 'click' and click_name = 'video_call' and p_product != 'manager' and pay_user = 1
and `date` >= '2023-06-04'
group by distinct_id, author_id_str) d
on a.distinct_id = d.distinct_id and a.author_id_str = d.author_id_str
left outer join
(select distinct_id, author_id_str, count(*) as cnt from events where
event = 'task' and task_name = 'video_chat_call_end' and p_product != 'manager'
and `date` >= '2023-06-04'
group by distinct_id, author_id_str) e
on a.distinct_id = e.distinct_id and a.author_id_str = e.author_id_str
left outer join
(select first_id, groupId from users) c
on a.distinct_id = c.first_id
where a.cnt >= 4
and b.cnt is null and d.cnt is null and e.cnt is null



select a.distinct_id, a.author_id_str, a.country, a.author_country, a.time from
(select distinct_id, author_id_str, duration_time, `time` as time, `$country` as country, author_country,
`$model` as model, `$manufacturer` as manufacturer, `$screen_height` as height, library from events where
event = 'task' and task_name = 'video_chat_call_end' and
((library = 'payZego' and duration_time > 56) or (library = 'blurVideo' and duration_time > 14)
or (library = 'freeVideo' and duration_time > 28) or (library = 'freeZego' and duration_time > 28) or (library = 'payVideo' and duration_time > 56))
and author_id_str is not null and author_id_str != ''
and `date` >='2023-07-01') a
join
(select distinct_id, max(pay_user) as pay_status from events where
event = 'task' and task_name = 'video_chat_call_end'
and `date` >= '2023-07-01'
group by distinct_id) b
on a.distinct_id = b.distinct_id
where b.pay_status = 1
