
select a.distinct_id,
       isnull(e.order_num, 0) as order_num,
       isnull(a.cnt, 0) as start_num,
       isnull(b.cnt, 0) as purchase_pop_num,
       isnull(c.cnt, 0) as pop_up_buy_num,
       isnull(f.cnt, 0) as chat_num,
       isnull(d.cnt, 0) as video_call_click_num,
       a.country, a.model, a.manufacturer, a.height,
       campaignId, groupId, isAccurateUser, isMolocoAdsUser

from
(select distinct_id, count(*) as cnt, max(`$country`) as country,
max(`$model`) as model, max(`$manufacturer`) as manufacturer, max(`$screen_height`) as height
from events
where `$is_first_day` = 1 and pay_user = 0
and event = '$AppStart'
and date >= '2023-10-01' and date < '2023-12-01'
group by distinct_id) a

left outer join
(select distinct_id, count(*) as cnt
from events
where `$is_first_day` = 1 and pay_user = 0
and event = 'task' and task_name = 'purchase_pop'
and date >= '2023-10-01' and date < '2023-12-01'
group by distinct_id) b
on a.distinct_id = b.distinct_id

left outer join
(select distinct_id, count(*) as cnt
from events
where `$is_first_day` = 1 and pay_user = 0
and event = 'task' and task_name = 'pop_up_buy'
and date >= '2023-10-01' and date < '2023-12-01'
group by distinct_id) c
on a.distinct_id = c.distinct_id

left outer join
(select distinct_id, count(*) as cnt
from events
where `$is_first_day` = 1 and pay_user = 0
and event = 'task' and task_name = 'chat'
and date >= '2023-10-01' and date < '2023-12-01'
group by distinct_id) f
on a.distinct_id = f.distinct_id

left outer join
(select distinct_id, count(*) as cnt
from events
where `$is_first_day` = 1 and pay_user = 0
and event = 'click' and click_name = 'video_call'
and date >= '2023-10-01' and date < '2023-12-01'
group by distinct_id) d
on a.distinct_id = d.distinct_id

left outer join
(select distinct_id, max(order_num) as order_num, max(pay_user) as pay_user
from events
where event = 'task'
and date >= '2023-10-01'
group by distinct_id) e
on a.distinct_id = e.distinct_id

left outer join
(select first_id, campaignId, groupId, isAccurateUser, isGoogleAdsUser, isFbAdsUser, isMolocoAdsUser, networkType from users) h
on a.distinct_id = h.first_id

order by isnull(e.order_num, 0) desc