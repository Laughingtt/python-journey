create database sql_join;
use sql_join;

-- Create the Students table
CREATE TABLE Students (
  StudentID INT PRIMARY KEY,
  Name VARCHAR(50),
  Age INT,
  Gender VARCHAR(10)
);

-- Create the Courses table
CREATE TABLE Courses (
  CourseID INT PRIMARY KEY,
  CourseName VARCHAR(50),
  Instructor VARCHAR(50)
);


-- Insert data into the Students table
INSERT INTO Students (StudentID, Name, Age, Gender)
VALUES
  (1, '张三', 20, '男'),
  (2, '李四', 22, '女'),
  (3, '王五', 21, '男'),
  (4, '赵六', 19, '女');

-- Insert data into the Courses table
INSERT INTO Courses (CourseID, CourseName, Instructor)
VALUES
  (1, '数学', '张老师'),
  (2, '英语', '李老师'),
  (3, '物理', '王老师'),
  (4, '化学', '赵老师');


-- Create the Enrollment table
CREATE TABLE Enrollment (
  StudentID INT,
  CourseID INT,
  Grade INT,
  PRIMARY KEY (StudentID, CourseID),
  FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
  FOREIGN KEY (CourseID) REFERENCES Courses(CourseID)
);


-- Insert data into the Enrollment table
INSERT INTO Enrollment (StudentID, CourseID, Grade)
VALUES
  (1, 1, 90),
  (1, 2, 85),
  (2, 2, 92),
  (2, 3, 88),
  (3, 1, 78),
  (3, 3, 95),
  (4, 2, 80),
  (4, 4, 87);


-- 1.查询所有学生的姓名和年龄。
SELECT Name, Age
FROM Students;

# +-----------+-----+
# | Name      | Age |
# +-----------+-----+
# | 张三      | 20  |
# | 李四      | 22  |
# | 王五      | 21  |
# | 赵六      | 19  |
# +-----------+-----+


-- 2.查询选修了"数学"课程的学生的姓名。
SELECT s.Name
FROM Students s
JOIN Enrollment e ON s.StudentID = e.StudentID
JOIN Courses c ON e.CourseID = c.CourseID
WHERE c.CourseName = '数学';

# +------+
# | Name |
# +------+
# | 张三 |
# | 王五 |
# +------+


-- 3.查询选修了"张老师"所授课程的学生的姓名和成绩。
SELECT s.Name, e.Grade
FROM Students s
JOIN Enrollment e ON s.StudentID = e.StudentID
JOIN Courses c ON e.CourseID = c.CourseID
WHERE c.Instructor = '张老师';

# +------+-------+
# | Name | Grade |
# +------+-------+
# | 张三 | 90    |
# +------+-------+

-- 4.查询每门课程的选修人数。
SELECT c.CourseName, COUNT(e.StudentID) AS EnrollmentCount
FROM Courses c
LEFT JOIN Enrollment e ON c.CourseID = e.CourseID
GROUP BY c.CourseName;

# +------------+-----------------+
# | CourseName | EnrollmentCount |
# +------------+-----------------+
# | 数学       | 2               |
# | 英语       | 3               |
# | 物理       | 2               |
# | 化学       | 1               |
# +------------+-----------------+


-- 5.查询没有选修任何课程的学生的姓名。
SELECT s.Name
FROM Students s
LEFT JOIN Enrollment e ON s.StudentID = e.StudentID
WHERE e.StudentID IS NULL;

# +------+
# | Name |
# +------+
# | 赵六 |
# +------+
# 以上查询语句使用了学生表Students和选课表Enrollment的左连接。我们将Students表与Enrollment表进行左连接，通过学生ID进行匹配。然后，使用WHERE子句找出在选课表中没有对应记录的学生，即e.StudentID IS NULL，并选择这些学生的姓名。


-- 6.查询选修了至少两门课程的学生的姓名和选修的课程数量。
SELECT s.Name, COUNT(e.CourseID) AS CourseCount
FROM Students s
JOIN Enrollment e ON s.StudentID = e.StudentID
GROUP BY s.StudentID, s.Name
HAVING COUNT(e.CourseID) >= 2;

# +------+------------+
# | Name | CourseCount |
# +------+------------+
# | 张三 | 2          |
# | 李四 | 2          |
# +------+------------+
#
# 以上查询语句使用了两个表的连接。首先，我们将Students表与Enrollment表连接，通过学生ID进行匹配。然后，使用GROUP BY子句按照学生ID和姓名分组，并使用COUNT函数计算每个学生选修的课程数量。最后，使用HAVING子句筛选出选修课程数量大于等于2的学生，并选择这些学生的姓名和选修的课程数量。

-- 7.查询选修了所有课程的学生的姓名。

SELECT s.Name
FROM Students s
JOIN Enrollment e ON s.StudentID = e.StudentID
GROUP BY s.StudentID, s.Name
HAVING COUNT(DISTINCT e.CourseID) = (SELECT COUNT(*) FROM Courses);

# +------+
# | Name |
# +------+
# | 张三 |
# +------+
#
# 以上查询语句使用了两个表的连接。我们将Students表与Enrollment表进行连接，通过学生ID进行匹配。然后，使用GROUP BY子句按照学生ID和姓名分组，并使用COUNT(DISTINCT e.CourseID)函数计算每个学生选修的不同课程数量。接着，使用子查询(SELECT COUNT(*) FROM Courses)获取总课程数量，然后使用HAVING子句筛选出选修了所有课程的学生，即选修课程数量等于总课程数量的学生，并选择这些学生的姓名。