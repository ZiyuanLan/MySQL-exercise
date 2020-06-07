
----------------------------------------------------------------
#项目介绍#
#My Sony会员，共分为注册会员、铜牌会员、银牌会员、金牌会员4个等级。会员的等级由"成长值"数值决定，成长值越高，会员等级越高#
#加入索尼会员可以享受会员尊享价，官方延保以及累积成长等多个激励。用户可以通过索尼中国在线商城，索尼产品，索尼app，微信等渠道注册#
----------------------------------------------------------------
#问题#
#1.不同注册渠道的会员，年龄和成长值有什么区别？#
SELECT channel, member_level, AVG(age)
FROM sony_members
GROUP BY channel;
----------------------------------------------------------------
#2.所提供数据中，销售额最好的产品分别是哪些？#
SELECT p.Base_category, p.Category, p.Product_id, SUM(s.amount) AS 'GMV'
FROM sony_sales s
INNER JOIN sony_product p
ON s.Product_ID = p.Product_id
GROUP BY p.Product_ID
ORDER BY 'GMV' DESC;
----------------------------------------------------------------
#3.会员和非会员在购买金额方面，有什么主要区别?#
SELECT
CASE
WHEN m.member_id IS NULL THEN '非会员'
ELSE '会员'
END AS is_member,
SUM(amount) AS 'total amount',
SUM(quant) AS 'total quant',
SUM(amount)/COUNT(distinct s.member_id) AS avg_amount, #人均销售额
SUM(quant)/COUNT(distinct s.member_id) AS avg_quant, #人均销售量
COUNT(*)/COUNT(distinct s.member_id) AS avg_orders #人均订单量
FROM
sony_sales s
LEFT JOIN sony_product p ON s.product_id = p.product_ID
LEFT JOIN sony_members m ON s.member_id = m.member_id
GROUP BY is_member;
---------------------------------------------------------------
#4.会员和非会员在购买产品类别方面，有什么主要区别?#
SELECT
CASE
WHEN m.member_id IS NULL THEN '非会员'
ELSE '会员'
END AS is_member,
base_category,
SUM(amount) AS 'total amount',
SUM(quant) AS 'total quant',
SUM(amount)/COUNT(distinct s.member_id) AS avg_amount, #人均销售额
SUM(quant)/COUNT(distinct s.member_id) AS avg_quant, #人均销售量
COUNT(*)/COUNT(distinct s.member_id) AS avg_orders #人均订单量
FROM
sony_sales s
LEFT JOIN sony_product p ON s.product_id = p.product_ID
LEFT JOIN sony_members m ON s.member_id = m.member_id
GROUP BY is_member, base_category;












