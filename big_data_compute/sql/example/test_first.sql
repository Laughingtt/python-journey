-- 创建一个表格，包含以下字段：学生ID、学生姓名、科目、成绩
# 创建数据库
create database sql50;
use sql50;
CREATE TABLE students (
  student_id INT,
  student_name VARCHAR(50),
  subject VARCHAR(50),
  score INT
);

-- 将一些数据插入该表格中
INSERT INTO students (student_id, student_name, subject, score) VALUES
  (1, '张三', '语文', 90),
  (1, '张三', '数学', 85),
  (1, '张三', '英语', 92),
  (2, '李四', '语文', 80),
  (2, '李四', '数学', 80),
  (2, '李四', '英语', 85),
  (3, '王五', '语文', 75),
  (3, '王五', '数学', 90),
  (3, '王五', '英语', 80);

-- 通过SQL查询，列出所有学生的ID和姓名
SELECT student_id, student_name FROM students;

-- 通过SQL查询，列出所有科目的名称
SELECT DISTINCT subject FROM students;

-- 通过SQL查询，列出所有成绩大于80分的学生的ID和姓名
SELECT student_id, student_name FROM students WHERE score > 80;

-- 通过SQL查询，列出所有学生的ID、姓名、科目和成绩。并按成绩降序排列
SELECT student_id, student_name, subject, score FROM students ORDER BY score DESC;

-- 通过SQL查询，列出所有学生的ID、姓名和科目，但只显示成绩大于80分的学生
SELECT student_id, student_name, subject FROM students WHERE score > 80;

-- 通过SQL查询，列出所有学生的ID和姓名，并计算每个学生的平均成绩
SELECT student_id, student_name, AVG(score) AS avg_score FROM students GROUP BY student_id, student_name;

-- 通过SQL查询，列出所有学生的ID、姓名和科目，但只显示没有出现过80分以上的成绩的学生
SELECT student_id, student_name, subject FROM students WHERE score < 80;

-- 通过SQL查询，列出所有学生的ID和姓名，并计算每个学生的最高成绩
SELECT student_id, student_name, MAX(score) AS max_score FROM students GROUP BY student_id, student_name;

-- 查找不同科目的数量
SELECT COUNT(DISTINCT subject) AS subject_count FROM students;

-- 查找学生ID的数量
SELECT COUNT(DISTINCT student_id) AS student_id_count FROM students;

-- 查找平均成绩大于80分的学生ID和他们的平均成绩
SELECT student_id, AVG(score) AS avg_score FROM students WHERE score > 80 GROUP BY student_id;

-- 查找平均成绩最高的前5%的学生ID和他们的平均成绩
SELECT student_id, AVG(score) AS avg_score FROM students GROUP BY student_id ORDER BY AVG(score) DESC LIMIT 5;

-- 查找所有学生的ID、姓名和科目，并按照平均成绩降序排列
SELECT student_id, student_name, AVG(score) AS avg_score FROM students GROUP BY student_id,student_name ORDER BY avg_score DESC;

-- 查找所有学生的ID、姓名和科目，并按照最高成绩升序排列
SELECT student_id, student_name, MAX(score) AS max_score FROM students GROUP BY student_id,student_name ORDER BY max_score ASC;

-- 查找所有学生的ID、姓名和科目，并计算每个学生的最高成绩和平均成绩之间的差值
SELECT student_id, student_name, MAX(score) - AVG(score) AS diff FROM students GROUP BY student_id,student_name;