
--------------------------------------------------------------
#第一章 查询销售数据#

#找出销售产品的配送城市（只显示前10行）#
SELECT delivery_city FROM sales LIMIT 10;

#统计配送城市数量，去除重复记录#
SELECT COUNT(DISTINCT delivery_city) FROM sales;

#找出每笔销售记录销售数量的最小值和最大值，并求销售数量的总和#
SELECT MAX(quant), MIN(quant), SUM(quant) FROM sales;

#销售表中找出交易金额的平均值、总和、以及交易城市的个数（去重）#
SELECT AVG(amount), SUM(amount), COUNT(DISTINCT city) FROM sales;

-------------------------------------------------------------

#第二章 筛选会员/产品信息#

#从销售表中筛选出交易金额在100~500元之间的产品ID和交易金额# （精确匹配）
SELECT product_id, amount FROM sales 
where amount between 100 and 500;

#从销售表中，找出销售数量在5件以上同时配送城市为武汉的会员ID#

SELECT member_id from sales 

WHERE quant >5 AND delivery_city = "wuhan";



#销售表中筛选出武汉和上海的会员ID# (范围匹配)

select member_id from sales 

where city in ("wuhan","shanghai");



#从会员表中筛选出手机号码最后含09以及姓张的会员ID、手机号码和姓名#

SELECT member_id, phone, name FROM members 

WHERE phone LIKE '%09' AND name LIKE '张%'



#请从销售表中，找出南京白下区交易金额大于1000的门店名称以及交易金额# （模糊匹配）

SELECT shop_name, amount FROM sales 

WHERE shop_name LIKE "南京白下区%" AND amount > 1000;



#将销售表中的交易金额/交易量定义为单价#

SELECT amount/quant AS price FROM sales;



------------------------------------------------------------



#第三章 表连接#



(INNER场景：找到会员的消费金额)

#使用会员ID链接销售表与会员表，找出购买产品的会员ID以及消费金额#

SELECT m.member_id, s.amount

FROM sales s

INNER JOIN members m

ON s.member_id=m.member_id;



(RIGHT场景：找到会员与非会员的交易金额)

#使用会员表中的member_id匹配销售表中的member_id，查询购买产品的会员ID以及购买金额。（如果为非会员，则member_id为空值null）#

SELECT m.member_id, s.amount

FROM members m

RIGHT JOIN sales s

ON m.member_id=s.member_id;



(LEFT场景：找到未购买产品的会员)

#找到未购买产品的会员，他的名字，年龄，性别#

SELECT name, age, gender

FROM members m

LEFT JOIN sales s

on m.member_id = s.member_id

WHERE amount is NULL;



---------------------------------------------------------



#第四章 数据分组和排序#



#从销售表中找到业绩最好的门店#

SELECT shop_name, SUM(amount) AS total

FROM sales

Group BY shop_name

ORDER BY total;



#将会员的性别改为中文#

SELECT

CASE gender

WHEN 'm'THEN'男'

WHEN 'w'THEN'女'

END AS flag

FROM members;



#找出销售金额最高的产品大类与产品小类# (HAVING / WHERE 用法)

SELECT base_category, category, SUM(amount) AS total

FROM sales s

INNER JOIN product p

ON s.product_id=p.product_id

GROUP BY base_category, category

HAVING total > 250000

ORDER BY total DESC;



SELECT base_category, category, SUM(amount) AS total

FROM sales s

INNER JOIN product p

ON s.product_id=p.product_id

WHERE amount >= 5000

GROUP BY base_category, category

HAVING total > 250000

ORDER BY total DESC;



#找出购买产品和未购买产品的会员年龄和性别分布#

SELECT m.member_id, age, gender,

CASE

WHEN s.member_id IS NOT NULL THEN 'Y' ELSE 'N'

END AS flag

FROM sales s

RIGHT JOIN members m

ON m.member_id=s.member_id;



#找到不同年龄段的顾客平均消费金额的差异#

SELECT AVG(amount),

CASE

WHEN age<=35 THEN '青年'

WHEN age>=36 AND age<=50 THEN '中年'

WHEN age>51 THEN '老年'

END AS flag

FROM sales s

INNER JOIN members m

ON s.member_id=m.member_id

GROUP BY flag;



-----------------------------------------------------
