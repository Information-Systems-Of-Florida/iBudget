/*CREATE NONCLUSTERED INDEX IX_Consumers_Status_CaseNo 
ON tbl_Consumers(Status, CASENO) INCLUDE (DOB, RESIDENCETYPE);

CREATE NONCLUSTERED INDEX IX_Claims_CaseNo_ServiceDate 
ON tbl_Claims_MMIS(CaseNo, ServiceDate) INCLUDE (PaidAmt);

CREATE NONCLUSTERED INDEX IX_QSI_CaseNo_ApproveDate 
ON tbl_QSIAssessments(CASENO, APPROVEDATE) WHERE STATUS = 'Complete';
*/

/*
================================================================================
Stored Procedure: sp_GetiBudgetModelData
Purpose: Extract standardized dataset for all iBudget models (1-10)
Database: APD
Author: iBudget Algorithm Study Team
Created: 2024
================================================================================
This procedure returns all data needed for the 10 alternative models.
Individual models can select which columns to use.
================================================================================
*/

CREATE OR ALTER PROCEDURE [dbo].[sp_GetiBudgetModelData]
    @FiscalYearStart INT = 2019,
    @FiscalYearEnd INT = 2021,
    @IncludeDetailedDiagnostics BIT = 1
AS
BEGIN
    SET NOCOUNT ON;
    
    -- Declare fiscal year boundaries
    DECLARE @FY_Table TABLE (
        FiscalYear INT,
        StartDate DATE,
        EndDate DATE
    );
    
    -- Populate fiscal years (Sept 1 - Aug 31)
    DECLARE @Year INT = @FiscalYearStart;
    WHILE @Year <= @FiscalYearEnd
    BEGIN
        INSERT INTO @FY_Table (FiscalYear, StartDate, EndDate)
        VALUES (
            @Year,
            CAST(CONCAT(@Year - 1, '-09-01') AS DATE),
            CAST(CONCAT(@Year, '-08-31') AS DATE)
        );
        SET @Year = @Year + 1;
    END;
    
    -- CTE for Consumer-Year combinations
    WITH ConsumerYearBase AS (
        SELECT 
            c.CASENO,
            c.Id as ConsumerID,
            fy.FiscalYear,
            fy.StartDate as FY_StartDate,
            fy.EndDate as FY_EndDate,
            c.DOB,
            DATEDIFF(YEAR, c.DOB, fy.StartDate) as Age,
            c.GENDER,
            c.RACE,
            c.ETHNICITYLOOKUP as Ethnicity,
            c.RESIDENCETYPE,
            c.County,
            c.District,
            c.Region,
            c.PrimaryDiagnosis,
            c.SecondaryDiagnosis,
            c.OtherDiagnosis,
            c.MentalHealthDiag1,
            c.MentalHealthDiag2,
            c.DevelopmentalDisability,
            c.OPENDATE,
            c.CLOSEDATE,
            c.MedicaidId,
            c.PRIMARYWORKER,
            c.CBCFlag,
            c.Competency,
            
            -- Calculate days in system during fiscal year
            CASE 
                WHEN c.OPENDATE > fy.EndDate OR 
                     (c.CLOSEDATE IS NOT NULL AND c.CLOSEDATE < fy.StartDate) 
                THEN 0
                ELSE DATEDIFF(DAY,
                    CASE 
                        WHEN c.OPENDATE > fy.StartDate THEN c.OPENDATE 
                        ELSE fy.StartDate 
                    END,
                    CASE 
                        WHEN c.CLOSEDATE IS NULL OR c.CLOSEDATE > fy.EndDate THEN fy.EndDate 
                        ELSE c.CLOSEDATE 
                    END
                ) + 1
            END as DaysInSystem,
            
            -- Entry/Exit timing flags
            CASE 
                WHEN c.OPENDATE > fy.StartDate AND 
                     DATEDIFF(DAY, fy.StartDate, c.OPENDATE) > 30 
                THEN 1 ELSE 0 
            END as LateEntry,
            
            CASE 
                WHEN c.CLOSEDATE IS NOT NULL AND 
                     c.CLOSEDATE < fy.EndDate AND
                     DATEDIFF(DAY, c.CLOSEDATE, fy.EndDate) > 30 
                THEN 1 ELSE 0 
            END as EarlyExit
            
        FROM tbl_Consumers c
        CROSS JOIN @FY_Table fy
        WHERE c.Status = 'ACTIVE' 
            OR (c.Status = 'CLOSED' AND c.CLOSEDATE >= fy.StartDate)
    ),
    
    -- CTE for QSI Assessment Data
    QSIAssessmentData AS (
        SELECT 
            q.CASENO,
            q.AssessID,
            q.APPROVEDATE,
            q.STATUS as QSI_Status,
            
            -- Individual QSI Questions (Q14-Q35 are primary assessment items)
            ISNULL(TRY_CAST(q.Q14 as INT), 0) as Q14,  -- Vision
            ISNULL(TRY_CAST(q.Q15 as INT), 0) as Q15,  -- Hearing
            ISNULL(TRY_CAST(q.Q16 as INT), 0) as Q16,  -- Eating
            ISNULL(TRY_CAST(q.Q17 as INT), 0) as Q17,  -- Ambulation
            ISNULL(TRY_CAST(q.Q18 as INT), 0) as Q18,  -- Transfers
            ISNULL(TRY_CAST(q.Q19 as INT), 0) as Q19,  -- Toileting
            ISNULL(TRY_CAST(q.Q20 as INT), 0) as Q20,  -- Hygiene
            ISNULL(TRY_CAST(q.Q21 as INT), 0) as Q21,  -- Dressing
            ISNULL(TRY_CAST(q.Q22 as INT), 0) as Q22,  -- Communications
            ISNULL(TRY_CAST(q.Q23 as INT), 0) as Q23,  -- Self-Protection
            ISNULL(TRY_CAST(q.Q24 as INT), 0) as Q24,  -- Evacuation ability
            
            ISNULL(TRY_CAST(q.Q25 as INT), 0) as Q25,  -- Hurtful to Self/Self Injurious
            ISNULL(TRY_CAST(q.Q26 as INT), 0) as Q26,  -- Aggressive/Hurtful to Others
            ISNULL(TRY_CAST(q.Q27 as INT), 0) as Q27,  -- Destructive to Property
            ISNULL(TRY_CAST(q.Q28 as INT), 0) as Q28,  -- Inappropriate Sexual Behavior
            ISNULL(TRY_CAST(q.Q29 as INT), 0) as Q29,  -- Running Away
            ISNULL(TRY_CAST(q.Q30 as INT), 0) as Q30,  -- Other Behaviors
            
            ISNULL(TRY_CAST(q.Q31a as INT), 0) as Q31a,
            ISNULL(TRY_CAST(q.Q31b as INT), 0) as Q31b,
            ISNULL(TRY_CAST(q.Q32 as INT), 0) as Q32,  -- Injury from Self-Injurious
            ISNULL(TRY_CAST(q.Q33 as INT), 0) as Q33,  -- Injury from Aggression
            ISNULL(TRY_CAST(q.Q34 as INT), 0) as Q34,  -- Mechanical Restraints
            ISNULL(TRY_CAST(q.Q35 as INT), 0) as Q35,  -- Emergency Chemical Restraint
            
            -- Additional QSI Questions (Q36-Q50)
            ISNULL(TRY_CAST(q.Q36 as INT), 0) as Q36,  -- Psychotropic Medications
            ISNULL(TRY_CAST(q.Q37 as INT), 0) as Q37,  -- Gastrointestinal Conditions
            ISNULL(TRY_CAST(q.Q38 as INT), 0) as Q38,  -- Seizures
            ISNULL(TRY_CAST(q.Q39 as INT), 0) as Q39,  -- Anti-Epileptic Medication
            ISNULL(TRY_CAST(q.Q40 as INT), 0) as Q40,  -- Skin Breakdown
            ISNULL(TRY_CAST(q.Q41 as INT), 0) as Q41,  -- Bowel Function
            ISNULL(TRY_CAST(q.Q42 as INT), 0) as Q42,  -- Nutrition
            ISNULL(TRY_CAST(q.Q43 as INT), 0) as Q43,  -- Physician Treatment
            ISNULL(TRY_CAST(q.Q44 as INT), 0) as Q44,  -- Chronic Healthcare Needs
            ISNULL(TRY_CAST(q.Q45 as INT), 0) as Q45,  -- Individual's Injuries
            ISNULL(TRY_CAST(q.Q46 as INT), 0) as Q46,  -- Falls
            ISNULL(TRY_CAST(q.Q47 as INT), 0) as Q47,  -- Physician Visits/Nursing
            ISNULL(TRY_CAST(q.Q48 as INT), 0) as Q48,  -- Emergency Room Visits
            ISNULL(TRY_CAST(q.Q49 as INT), 0) as Q49,  -- Hospital Admission
            ISNULL(TRY_CAST(q.Q50 as INT), 0) as Q50,  -- Days Missed
            
            -- Calculate Functional Sum (Q14-Q24)
            (ISNULL(TRY_CAST(q.Q14 as INT), 0) + ISNULL(TRY_CAST(q.Q15 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q16 as INT), 0) + ISNULL(TRY_CAST(q.Q17 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q18 as INT), 0) + ISNULL(TRY_CAST(q.Q19 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q20 as INT), 0) + ISNULL(TRY_CAST(q.Q21 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q22 as INT), 0) + ISNULL(TRY_CAST(q.Q23 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q24 as INT), 0)) as FSum,
             
            -- Calculate Behavioral Sum (Q25-Q30)
            (ISNULL(TRY_CAST(q.Q25 as INT), 0) + ISNULL(TRY_CAST(q.Q26 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q27 as INT), 0) + ISNULL(TRY_CAST(q.Q28 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q29 as INT), 0) + ISNULL(TRY_CAST(q.Q30 as INT), 0)) as BSum,
             
            -- Calculate Physical Sum (Q32-Q50)
            (ISNULL(TRY_CAST(q.Q32 as INT), 0) + ISNULL(TRY_CAST(q.Q33 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q34 as INT), 0) + ISNULL(TRY_CAST(q.Q35 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q36 as INT), 0) + ISNULL(TRY_CAST(q.Q37 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q38 as INT), 0) + ISNULL(TRY_CAST(q.Q39 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q40 as INT), 0) + ISNULL(TRY_CAST(q.Q41 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q42 as INT), 0) + ISNULL(TRY_CAST(q.Q43 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q44 as INT), 0) + ISNULL(TRY_CAST(q.Q45 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q46 as INT), 0) + ISNULL(TRY_CAST(q.Q47 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q48 as INT), 0) + ISNULL(TRY_CAST(q.Q49 as INT), 0) + 
             ISNULL(TRY_CAST(q.Q50 as INT), 0)) as PSum,
             
            -- QSI Level Scores
            q.FLEVEL,  -- Functional Level
            q.BLEVEL,  -- Behavioral Level  
            q.PLEVEL,  -- Physical Level
            q.OLEVEL,  -- Overall Level
            q.LOSRI,   -- Level of Support Rating
            
            ROW_NUMBER() OVER (PARTITION BY q.CASENO ORDER BY q.APPROVEDATE DESC) as QSI_Rank
            
        FROM tbl_QSIAssessments q
        WHERE q.STATUS = 'Complete'
    ),
    
    -- CTE for Claims/Cost Data
    ClaimsAggregated AS (
        SELECT 
            cl.CaseNo,
            fy.FiscalYear,
            SUM(cl.PaidAmt) as TotalPaidClaims,
            COUNT(DISTINCT cl.ServiceDate) as ServiceDays,
            COUNT(DISTINCT cl.ProcCode) as UniqueProcedures,
            COUNT(DISTINCT cl.ProviderMedcId) as UniqueProviders,
            MIN(cl.ServiceDate) as FirstServiceDate,
            MAX(cl.ServiceDate) as LastServiceDate,
            SUM(CASE WHEN cl.PaidAmt > 0 THEN cl.PaidAmt ELSE 0 END) as PositivePaidAmount,
            SUM(CASE WHEN cl.PaidAmt < 0 THEN cl.PaidAmt ELSE 0 END) as NegativeAdjustments,
            COUNT(*) as TotalClaimLines
            
        FROM tbl_Claims_MMIS cl
        INNER JOIN @FY_Table fy ON 
            cl.ServiceDate >= fy.StartDate AND 
            cl.ServiceDate <= fy.EndDate
        GROUP BY cl.CaseNo, fy.FiscalYear
    ),
    
    -- CTE for Budget Data (if needed by some models)
    BudgetData AS (
        SELECT 
            b.CaseNo,
            b.FiscalYear,
            b.BudgetAmount,
            b.AnnualizedAmount,
            b.AmountEncumbered,
            b.BudgetType,
            b.BudgetStatus
        FROM tbl_Budgets b
        WHERE b.BudgetStatus ='Budget Approved'
    ),
    
    -- CTE to identify consumers with multiple QSI assessments in a fiscal year
    MultipleQSIFlags AS (
        SELECT 
            q.CASENO,
            fy.FiscalYear,
            COUNT(DISTINCT q.AssessID) as QSI_Count,
            CASE WHEN COUNT(DISTINCT q.AssessID) > 1 THEN 1 ELSE 0 END as HasMultipleQSI
        FROM tbl_QSIAssessments q
        INNER JOIN @FY_Table fy ON 
            q.APPROVEDATE >= DATEADD(MONTH, -3, fy.StartDate) AND 
            q.APPROVEDATE <= DATEADD(MONTH, 3, fy.EndDate)
        WHERE q.STATUS = 'Complete'
        GROUP BY q.CASENO, fy.FiscalYear
    )
    
    -- Main SELECT combining all data
    SELECT 
        -- Consumer Demographics
        cyb.CASENO,
        cyb.ConsumerID,
        cyb.FiscalYear,
        cyb.Age,
        cyb.GENDER,
        cyb.RACE,
        cyb.Ethnicity,
        cyb.County,
        cyb.District,
        cyb.Region,
        
        -- Diagnosis Information
        cyb.PrimaryDiagnosis,
        cyb.SecondaryDiagnosis,
        cyb.OtherDiagnosis,
        cyb.MentalHealthDiag1,
        cyb.MentalHealthDiag2,
        cyb.DevelopmentalDisability,
        
        -- Living Setting (Original)
        cyb.RESIDENCETYPE,
        
        -- Aggregated Living Setting (6 levels for models)
        CASE 
            WHEN cyb.RESIDENCETYPE = 'FH' OR cyb.RESIDENCETYPE LIKE 'Family%' 
                THEN 'FH'
            WHEN cyb.RESIDENCETYPE IN ('ILSL', 'NC', 'LTRC', 'GH') OR 
                 cyb.RESIDENCETYPE LIKE 'Independent%' OR 
                 cyb.RESIDENCETYPE LIKE 'Supported%' OR
                 cyb.RESIDENCETYPE LIKE 'Group%'
                THEN 'ILSL'
            WHEN cyb.RESIDENCETYPE IN ('RH1', 'RH2', 'RH3', 'RH4', 'RH5', 'RHLI') OR
                 cyb.RESIDENCETYPE LIKE 'Residential Habilitation%Standard%' OR
                 cyb.RESIDENCETYPE LIKE 'Residential Habilitation%Basic%' OR
                 cyb.RESIDENCETYPE LIKE 'Residential Habilitation%Minimal%' OR
                 cyb.RESIDENCETYPE LIKE 'Residential Habilitation%Moderate%' OR
                 cyb.RESIDENCETYPE LIKE 'Residential Habilitation%Extensive 1%' OR
                 cyb.RESIDENCETYPE LIKE 'Residential Habilitation%Extensive 2%' OR
                 cyb.RESIDENCETYPE LIKE 'Residential Habilitation%Live In%'
                THEN 'RH1'
            WHEN cyb.RESIDENCETYPE LIKE '%Behavior Focus%' OR 
                 cyb.RESIDENCETYPE LIKE '%Behavioral Focus%' OR
                 cyb.RESIDENCETYPE IN ('RHBF1', 'RHBF2', 'RHBF3', 'RHBF4') 
                THEN 'RH2'
            WHEN cyb.RESIDENCETYPE LIKE '%Intensive Behavior%' OR 
                 cyb.RESIDENCETYPE LIKE '%Intensive Behavioral%' OR
                 cyb.RESIDENCETYPE IN ('RHIB1', 'RHIB2', 'RHIB3', 'RHIB4') 
                THEN 'RH3'
            WHEN cyb.RESIDENCETYPE LIKE '%CTEP%' OR 
                 cyb.RESIDENCETYPE LIKE '%Special Medical%' OR
                 cyb.RESIDENCETYPE = 'SHC' OR
                 cyb.RESIDENCETYPE IN ('RHCTEP1', 'RHCTEP2', 'RHCTEP3', 'RHCTEP4') 
                THEN 'RH4'
            ELSE 'FH'  -- Default to Family Home if not classified
        END as LivingSetting,
        
        -- Age Groups (3 categories)
        CASE 
            WHEN cyb.Age <= 20 THEN 'Age3_20'
            WHEN cyb.Age BETWEEN 21 AND 30 THEN 'Age21_30'
            ELSE 'Age31Plus'
        END as AgeGroup,
        
        -- QSI Assessment Data (all questions)
        qsi.Q14, qsi.Q15, qsi.Q16, qsi.Q17, qsi.Q18, qsi.Q19, qsi.Q20,
        qsi.Q21, qsi.Q22, qsi.Q23, qsi.Q24, qsi.Q25, qsi.Q26, qsi.Q27,
        qsi.Q28, qsi.Q29, qsi.Q30, qsi.Q31a, qsi.Q31b, qsi.Q32, qsi.Q33,
        qsi.Q34, qsi.Q35, qsi.Q36, qsi.Q37, qsi.Q38, qsi.Q39, qsi.Q40,
        qsi.Q41, qsi.Q42, qsi.Q43, qsi.Q44, qsi.Q45, qsi.Q46, qsi.Q47,
        qsi.Q48, qsi.Q49, qsi.Q50,
        
        -- QSI Summary Scores
        qsi.FSum,  -- Functional Sum (Q14-Q24)
        qsi.BSum,  -- Behavioral Sum (Q25-Q30)
        qsi.PSum,  -- Physical Sum (Q32-Q50)
        
        -- QSI Levels
        qsi.FLEVEL,
        qsi.BLEVEL,
        qsi.PLEVEL,
        qsi.OLEVEL,
        qsi.LOSRI,
        
        -- Cost/Claims Data
        ISNULL(cl.TotalPaidClaims, 0) as TotalCost,
        ISNULL(cl.PositivePaidAmount, 0) as PositiveCost,
        ISNULL(cl.NegativeAdjustments, 0) as Adjustments,
        ISNULL(cl.ServiceDays, 0) as ServiceDays,
        ISNULL(cl.UniqueProcedures, 0) as UniqueProcedures,
        ISNULL(cl.UniqueProviders, 0) as UniqueProviders,
        ISNULL(cl.TotalClaimLines, 0) as ClaimLines,
        
        -- Budget Data (optional for some models)
        ISNULL(bd.BudgetAmount, 0) as BudgetAmount,
        ISNULL(bd.AnnualizedAmount, 0) as AnnualizedBudget,
        
        -- System Participation
        cyb.DaysInSystem,
        cyb.LateEntry,
        cyb.EarlyExit,
        
        -- Data Quality Flags
        CASE WHEN qsi.CASENO IS NULL THEN 1 ELSE 0 END as MissingQSI,
        ISNULL(mq.HasMultipleQSI, 0) as HasMultipleQSI,
        CASE WHEN ISNULL(cl.TotalPaidClaims, 0) <= 0 THEN 1 ELSE 0 END as NoPositiveCost,
        CASE WHEN cyb.DaysInSystem < 30 THEN 1 ELSE 0 END as InsufficientDays,
        
        -- Usability Flag (for exclusion criteria)
        CASE 
            WHEN cyb.DaysInSystem >= 30 AND
                 cyb.LateEntry = 0 AND
                 cyb.EarlyExit = 0 AND
                 qsi.CASENO IS NOT NULL AND
                 ISNULL(mq.HasMultipleQSI, 0) = 0 AND
                 ISNULL(cl.TotalPaidClaims, 0) > 0 AND
                 ISNULL(cl.ServiceDays, 0) >= 30
            THEN 1 
            ELSE 0 
        END as Usable,
        
        -- Additional metadata
        cyb.FY_StartDate,
        cyb.FY_EndDate,
        qsi.APPROVEDATE as QSI_ApprovalDate,
        cl.FirstServiceDate,
        cl.LastServiceDate
        
    FROM ConsumerYearBase cyb
    LEFT JOIN QSIAssessmentData qsi ON 
        cyb.CASENO = qsi.CASENO AND 
        qsi.QSI_Rank = 1  -- Most recent QSI
    LEFT JOIN ClaimsAggregated cl ON 
        cyb.CASENO = cl.CaseNo AND 
        cyb.FiscalYear = cl.FiscalYear
    LEFT JOIN BudgetData bd ON 
        cyb.CASENO = bd.CaseNo AND 
        CAST(cyb.FiscalYear as VARCHAR(10)) = bd.FiscalYear
    LEFT JOIN MultipleQSIFlags mq ON 
        cyb.CASENO = mq.CASENO AND 
        cyb.FiscalYear = mq.FiscalYear
    WHERE 
        cyb.DaysInSystem > 0  -- Consumer was in system during fiscal year
		and qsi.Q14 is not null
    ORDER BY 
        cyb.CASENO, 
        cyb.FiscalYear;
        
END;
GO

/*
================================================================================
USAGE EXAMPLES:
================================================================================

-- Get all data for fiscal years 2019-2021
EXEC sp_GetiBudgetModelData @FiscalYearStart = 2019, @FiscalYearEnd = 2021;

-- Get data for single fiscal year
EXEC sp_GetiBudgetModelData @FiscalYearStart = 2024, @FiscalYearEnd = 2025;

-- Get data without detailed diagnostics
EXEC sp_GetiBudgetModelData 
    @FiscalYearStart = 2019, 
    @FiscalYearEnd = 2021,
    @IncludeDetailedDiagnostics = 0;

================================================================================
OUTPUT COLUMNS:
================================================================================
The stored procedure returns 100+ columns including:
- Demographics (Age, Gender, Race, County, etc.)
- Living Setting (Original and 6-level aggregated)
- QSI Questions (Q14-Q50)
- QSI Summary Scores (FSum, BSum, PSum)
- Cost Data (TotalCost, ServiceDays, etc.)
- Data Quality Flags (Usable, MissingQSI, etc.)
- System Participation (DaysInSystem, LateEntry, EarlyExit)

Models can select which columns to use based on their requirements.
================================================================================
*/