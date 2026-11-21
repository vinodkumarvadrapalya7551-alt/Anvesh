# SkillBridge EdTech Academic Policies & Regulations Manual
**Effective Academic Year:** 2024–2025  
**Institutional Reference Benchmark Date:** January 15, 2025  

---

## 1. Institutional Calendar & Academic Timeline
1. **Academic Year Structure**: The academic year consists of two primary semesters (Fall and Spring) and an optional Summer term.
   - **Fall Semester 2024 Start Date**: September 1, 2024
   - **Census / Add-Drop Cutoff Date**: September 20, 2024
   - **Midterm Evaluation Window**: October 25, 2024 to October 31, 2024
   - **Spring Semester 2025 Start Date**: January 6, 2025
   - **Reference Date for Calculations**: **January 15, 2025**. All dynamic metrics, including student enrollment duration, elapsed weeks, attendance percentage to date, and active status, must strictly be computed against **January 15, 2025**.
2. **Elapsed Time Calculation Rule**:
   - Elapsed weeks from a start date \(D_{\text{start}}\) to the Reference Date (2025-01-15) are calculated as:
     $$\text{Weeks Elapsed} = \lfloor \frac{\text{Date(2025-01-15)} - D_{\text{start}}}{7} \rfloor$$
   - Elapsed days are calculated as the integer difference: $\text{Date(2025-01-15)} - D_{\text{start}}$.

---

## 2. Attendance Policies & Calculation Formula
1. **Components of Course Delivery**:
   Each course at SkillBridge EdTech comprises two pedagogical components:
   - **Lecture / Theory Sessions**: Conceptual foundations delivered synchronously or asynchronously.
   - **Lab / Practical / Tutorial Sessions**: Hands-on application, problem sets, and interactive coding.
2. **Weighted Attendance Formula**:
   Because practical mastery is paramount, attendance is evaluated with differential weighting:
   - **Lecture Attendance Weight**: 40% (0.40)
   - **Lab / Practical Attendance Weight**: 60% (0.60)
   - **Official Formula**:
     $$\text{Weighted Attendance (\%)} = (\text{Lecture Attendance \%} \times 0.40) + (\text{Lab Attendance \%} \times 0.60)$$
   - *Example*: If a student attends 80% of lectures and 70% of labs:
     $$\text{Weighted Attendance} = (80 \times 0.40) + (70 \times 0.60) = 32.0 + 42.0 = 74.0\%$$
3. **Attendance Status Categories & Thresholds**:
   - **Good Standing**: Weighted Attendance $\ge 75.0\%$. Student is fully eligible for final assessments and credentialing.
   - **Attendance Shortage / Warning**: Weighted Attendance between $65.0\%$ and $74.9\%$. The student receives an Attendance Warning. To sit for the final exam, the student must submit a condonation petition, and their maximum achievable course grade is capped at **B** (85.0%).
   - **Critical Debarment (Severe Shortage)**: Weighted Attendance $< 65.0\%$. The student is formally debarred from the final examination. Their enrollment status is updated to `Debarred` and the grade is recorded as `F` (0.0).

---

## 3. Grading Scheme & Course Performance
1. **Assessment Breakdown**:
   Student assessment is based on three components:
   - **Continuous Assessments (Quizzes / Homework)**: 30% total weight (0.30).
   - **Midterm Examination**: 30% total weight (0.30).
   - **Final Examination / Capstone Project**: 40% total weight (0.40).
2. **Composite Course Grade Formula**:
   $$\text{Final Score} = (\text{Continuous Score} \times 0.30) + (\text{Midterm Score} \times 0.30) + (\text{Final Exam Score} \times 0.40)$$
3. **Letter Grade & Grade Point Scale**:
   - **A (Outstanding)**: $90.0\% \le \text{Final Score} \le 100.0\%$ (Grade Point: 4.0)
   - **B (Above Average)**: $80.0\% \le \text{Final Score} < 90.0\%$ (Grade Point: 3.0)
   - **C (Satisfactory)**: $70.0\% \le \text{Final Score} < 80.0\%$ (Grade Point: 2.0)
   - **D (Pass / Minimum)**: $50.0\% \le \text{Final Score} < 70.0\%$ (Grade Point: 1.0)
   - **F (Fail)**: $\text{Final Score} < 50.0\%$ (Grade Point: 0.0)
4. **Dual-Passing Condition**:
   To pass any course, a student must simultaneously satisfy two criteria:
   1. The composite final score must be $\ge 50.0\%$.
   2. The score on the Final Exam / Capstone must be $\ge 40.0\%$.
   If the student achieves an overall score $\ge 50.0\%$ but scores $< 40.0\%$ on the final exam, the grade defaults to **D** (capped at 50.0%) or **F** if overall $< 50.0\%$.

---

## 4. Academic Early Warning & Probation Trigger System
SkillBridge EdTech utilizes an automated monitoring system to flag students requiring academic intervention as of the Reference Date (2025-01-15).

1. **Level 1: Academic Warning**:
   A student is placed on **Academic Warning** (Level 1 Alert) in a course if **ANY** of the following conditions are met:
   - Current Weighted Attendance is $< 75.0\%$.
   - Cumulative Continuous Assessment average to date is $< 50.0\%$.
   - Midterm examination score is $< 50.0\%$.

2. **Level 2: Critical Academic Probation**:
   A student is escalated to **Critical Academic Probation** (Level 2 Alert) if **ANY** of the following severe conditions occur:
   - Weighted Attendance is $< 60.0\%$.
   - Two or more submitted assessments or exams have scores $< 40.0\%$.
   - The student has $> 4$ consecutive weeks of recorded inactivity since enrollment date without an approved leave.
   - Cumulative GPA across all enrolled courses is $< 2.0$.

3. **Intervention Requirement**:
   Students with Level 1 Warning are assigned peer tutoring. Students on Level 2 Probation are required to undergo mandatory academic counseling and have their course load capped at 3 courses.

---

## 5. Course Completion & Progress Metrics
1. **Expected Course Duration**: Standard course duration is 16 weeks (112 days).
2. **Current Progress Calculation**:
   As of Reference Date (2025-01-15), for a student enrolled on date $D_{\text{enroll}}$:
   $$\text{Weeks Elapsed} = \lfloor \frac{\text{Date(2025-01-15)} - D_{\text{enroll}}}{7} \rfloor$$
   $$\text{Expected Progress (\%)} = \min\left(100.0, \frac{\text{Weeks Elapsed}}{16} \times 100\right)$$
3. **Pace Status**:
   - **Ahead of Schedule**: Completed modules exceed expected progress by $\ge 10\%$.
   - **On Track**: Completed modules within $\pm 10\%$ of expected progress.
   - **Behind Schedule**: Completed modules lag expected progress by $> 10\%$.

---

## 6. Scope Boundaries & Out-of-Scope Classification
The SkillBridge EdTech Academic System exclusively records and reasons over:
- Student enrollment data (Student ID, Name, Cohort, Enrollment Date).
- Course catalog metadata (Course ID, Course Title, Credits, Department).
- Attendance logs (Lectures attended/total, Labs attended/total).
- Assessment scores (Quizzes, Midterms, Final Projects, GPA).
- Institutional academic rules, formulas, and alert policies defined herein.

### Out-of-Scope Queries (Must return `answerable = 0`):
Any query requesting information outside these bounds CANNOT be answered by the system and must return `answerable: 0`. Examples include:
- Financial questions: Tuition fees, payment receipts, installment balances, billing disputes, or refunds.
- Extracurricular & athletic activities: University sports teams, club memberships, tournament rankings.
- Campus housing & accommodation: Dormitory assignments, cafeteria meal plans, room numbers.
- Personal contact & health records: Personal phone numbers, home addresses, private medical history, doctor notes.
- Career & placements: Post-graduation starting salaries, company job offers, external internship agreements.
- Unregistered entities: Queries referring to student IDs not present in the database or courses not offered in the catalog.
