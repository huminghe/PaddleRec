
select a.distinct_id, a.author_id_str, a.country, a.author_country, a.time,
c.campaignId, c.groupId, a.manufacturer, a.height, a.model, a.p_product from
(select distinct_id, author_id_str, duration_time, `time` as time, `$country` as country, author_country, p_product,
`$model` as model, `$manufacturer` as manufacturer, `$screen_height` as height from events where
event = 'click' and click_element = 'CARD' and page_url = 'meet'
and author_id_str is not null
and `date` >='2022-01-01' and `date` < '2023-01-01') a
join
(select distinct_id, min(`time`) as first_time from events where
event = 'task' and task_name = 'purchase_pop'
and task_result = 'success' and (order_num = 1 or order_num = 0) and isFirstBuy = 1
and `date` >= '2022-01-01'
group by distinct_id) b
on a.distinct_id = b.distinct_id
left outer join
(select first_id, groupId, campaignId from users) c
on a.distinct_id = c.first_id
where a.time < b.first_time


select a.distinct_id, a.author_id_str, a.country, a.author_country, a.time,
c.campaignId, c.groupId, a.manufacturer, a.height, a.model, a.p_product from
(select distinct_id, author_id_str, duration_time, `time` as time, `$country` as country, author_country, p_product,
`$model` as model, `$manufacturer` as manufacturer, `$screen_height` as height from events where
event = 'click' and click_name = 'LIKE'
and author_id_str is not null
and `date` >='2022-01-01') a
join
(select distinct_id, min(`time`) as first_time from events where
event = 'task' and task_name = 'purchase_pop'
and task_result = 'success' and (order_num = 1 or order_num = 0) and isFirstBuy = 1
and `date` >= '2022-01-01'
group by distinct_id) b
on a.distinct_id = b.distinct_id
left outer join
(select first_id, groupId, campaignId from users) c
on a.distinct_id = c.first_id
where a.time < b.first_time
