-- =============================================
-- Stored Procedure: sp_Outliers
-- Purpose: Analyze data quality and identify exclusions for iBudget model calibration
-- Use: exec sp_Outliers
-- =============================================

create PROCEDURE sp_Outliers
AS
BEGIN
    SET NOCOUNT ON;
    
    -- =============================================
    -- CONFIGURATION SECTION
    -- =============================================
    DECLARE @StartMonth INT = 9  -- September
    DECLARE @StartDay INT = 1
    
    -- =============================================
    -- Clean up any existing temp tables
    -- =============================================
    IF OBJECT_ID('tempdb..#FiscalYearClaims') IS NOT NULL
        DROP TABLE #FiscalYearClaims
    
    IF OBJECT_ID('tempdb..#ConsumerYearSummary') IS NOT NULL
        DROP TABLE #ConsumerYearSummary
    
    IF OBJECT_ID('tempdb..#MidYearQSI') IS NOT NULL
        DROP TABLE #MidYearQSI
    
    IF OBJECT_ID('tempdb..#ExclusionAnalysis') IS NOT NULL
        DROP TABLE #ExclusionAnalysis
    
    -- =============================================
    -- Assign fiscal years to all claims
    -- =============================================
    SELECT 
        c.CaseNo,
        c.ServiceDate,
        c.PaidAmt,
        CASE 
            WHEN MONTH(c.ServiceDate) >= @StartMonth 
            THEN YEAR(c.ServiceDate)
            ELSE YEAR(c.ServiceDate) - 1
        END AS FiscalYear,
        q.APPROVEDATE as QSI_Date,
        q.QSI_RowNum
    INTO #FiscalYearClaims
    FROM tbl_Claims_MMIS c
    LEFT JOIN (
        SELECT 
            CaseNo,
            APPROVEDATE,
            ROW_NUMBER() OVER (PARTITION BY CaseNo, APPROVEDATE ORDER BY APPROVEDATE) as QSI_RowNum,
            ROW_NUMBER() OVER (PARTITION BY CaseNo ORDER BY APPROVEDATE DESC) as rn
        FROM tbl_QSIAssessments
    ) q ON c.CaseNo = q.CaseNo 
        AND q.rn = 1
        AND q.APPROVEDATE <= c.ServiceDate
    
    -- =============================================
    -- Create consumer-fiscal year summary
    -- =============================================
    SELECT 
        CaseNo,
        FiscalYear,
        COUNT(DISTINCT ServiceDate) as ServiceDays,
        SUM(PaidAmt) as TotalPaidAmt,
        MIN(ServiceDate) as FirstServiceDate,
        MAX(ServiceDate) as LastServiceDate,
        DATEDIFF(DAY, 
            CONVERT(DATE, CAST(FiscalYear AS VARCHAR(4)) + '-09-01'),
            MIN(ServiceDate)) as DaysFromFYStart,
        DATEDIFF(DAY, 
            MAX(ServiceDate),
            DATEADD(DAY, -1, CONVERT(DATE, CAST(FiscalYear + 1 AS VARCHAR(4)) + '-09-01'))) as DaysToFYEnd,
        COUNT(DISTINCT QSI_Date) as QSI_Count
    INTO #ConsumerYearSummary
    FROM #FiscalYearClaims
    GROUP BY CaseNo, FiscalYear
    
    -- =============================================
    -- Identify mid-year QSI changes
    -- =============================================
    ;WITH QSI_FY AS (
        SELECT 
            CaseNo,
            APPROVEDATE,
            CASE 
                WHEN MONTH(APPROVEDATE) >= @StartMonth 
                THEN YEAR(APPROVEDATE)
                ELSE YEAR(APPROVEDATE) - 1
            END AS FiscalYear
        FROM tbl_QSIAssessments
    )
    SELECT 
        CaseNo,
        FiscalYear,
        COUNT(DISTINCT APPROVEDATE) as QSI_Changes_In_Year
    INTO #MidYearQSI
    FROM QSI_FY
    GROUP BY CaseNo, FiscalYear
    HAVING COUNT(DISTINCT APPROVEDATE) > 1
    
    -- =============================================
    -- Create exclusion flags
    -- =============================================
    SELECT 
        cys.*,
        CASE WHEN ISNULL(mq.QSI_Changes_In_Year, 0) > 1 THEN 1 ELSE 0 END as Flag_MidYearQSI,
        CASE WHEN cys.DaysFromFYStart > 30 THEN 1 ELSE 0 END as Flag_LateEntry,
        CASE WHEN cys.DaysToFYEnd > 30 THEN 1 ELSE 0 END as Flag_EarlyExit,
        CASE WHEN ISNULL(cys.TotalPaidAmt, 0) = 0 THEN 1 ELSE 0 END as Flag_NoCosts,
        CASE WHEN ISNULL(cys.TotalPaidAmt, 0) < 0 THEN 1 ELSE 0 END as Flag_NegativeCosts,
        CASE WHEN cys.ServiceDays < 30 THEN 1 ELSE 0 END as Flag_InsufficientService,
        CASE WHEN ISNULL(cys.QSI_Count, 0) = 0 THEN 1 ELSE 0 END as Flag_NoQSI
    INTO #ExclusionAnalysis
    FROM #ConsumerYearSummary cys
    LEFT JOIN #MidYearQSI mq 
        ON cys.CaseNo = mq.CaseNo 
        AND cys.FiscalYear = mq.FiscalYear
    
    -- =============================================
    -- RESULT SET 1: Overall Summary Statistics
    -- =============================================
    DECLARE @RecordCount INT
    SELECT @RecordCount = COUNT(*) FROM #ExclusionAnalysis
    
    SELECT 
        'Overall Statistics' as ReportSection,
        COUNT(DISTINCT CaseNo) as TotalUniqueConsumers,
        COUNT(*) as TotalConsumerYears,
        COUNT(DISTINCT FiscalYear) as TotalFiscalYears,
        MIN(FiscalYear) as EarliestFiscalYear,
        MAX(FiscalYear) as LatestFiscalYear,
        AVG(TotalPaidAmt) as AvgAnnualCost,
        STDEV(TotalPaidAmt) as StDevAnnualCost,
        MIN(TotalPaidAmt) as MinAnnualCost,
        MAX(TotalPaidAmt) as MaxAnnualCost,
        (SELECT MAX(TotalPaidAmt) FROM 
            (SELECT TOP 25 PERCENT TotalPaidAmt 
             FROM #ExclusionAnalysis 
             WHERE TotalPaidAmt IS NOT NULL
             ORDER BY TotalPaidAmt) AS Q1) as Q1_Cost,
        (SELECT MAX(TotalPaidAmt) FROM 
            (SELECT TOP 50 PERCENT TotalPaidAmt 
             FROM #ExclusionAnalysis 
             WHERE TotalPaidAmt IS NOT NULL
             ORDER BY TotalPaidAmt) AS Q2) as Median_Cost,
        (SELECT MAX(TotalPaidAmt) FROM 
            (SELECT TOP 75 PERCENT TotalPaidAmt 
             FROM #ExclusionAnalysis 
             WHERE TotalPaidAmt IS NOT NULL
             ORDER BY TotalPaidAmt) AS Q3) as Q3_Cost,
        (SELECT MIN(TotalPaidAmt) FROM 
            (SELECT TOP 5 PERCENT TotalPaidAmt 
             FROM #ExclusionAnalysis 
             WHERE TotalPaidAmt IS NOT NULL
             ORDER BY TotalPaidAmt DESC) AS P95) as P95_Cost,
        (SELECT MIN(TotalPaidAmt) FROM 
            (SELECT TOP 1 PERCENT TotalPaidAmt 
             FROM #ExclusionAnalysis 
             WHERE TotalPaidAmt IS NOT NULL
             ORDER BY TotalPaidAmt DESC) AS P99) as P99_Cost
    FROM #ExclusionAnalysis
    
    -- =============================================
    -- RESULT SET 2: Exclusion Summary by Reason
    -- =============================================
    SELECT 
        'Exclusion Summary' as ReportSection,
        SUM(Flag_MidYearQSI) as Count_MidYearQSI,
        CAST(100.0 * SUM(Flag_MidYearQSI) / NULLIF(@RecordCount, 0) AS DECIMAL(5,2)) as Pct_MidYearQSI,
        SUM(Flag_LateEntry) as Count_LateEntry,
        CAST(100.0 * SUM(Flag_LateEntry) / NULLIF(@RecordCount, 0) AS DECIMAL(5,2)) as Pct_LateEntry,
        SUM(Flag_EarlyExit) as Count_EarlyExit,
        CAST(100.0 * SUM(Flag_EarlyExit) / NULLIF(@RecordCount, 0) AS DECIMAL(5,2)) as Pct_EarlyExit,
        SUM(Flag_NoCosts) as Count_NoCosts,
        CAST(100.0 * SUM(Flag_NoCosts) / NULLIF(@RecordCount, 0) AS DECIMAL(5,2)) as Pct_NoCosts,
        SUM(Flag_NegativeCosts) as Count_NegativeCosts,
        CAST(100.0 * SUM(Flag_NegativeCosts) / NULLIF(@RecordCount, 0) AS DECIMAL(5,2)) as Pct_NegativeCosts,
        SUM(Flag_InsufficientService) as Count_InsufficientService,
        CAST(100.0 * SUM(Flag_InsufficientService) / NULLIF(@RecordCount, 0) AS DECIMAL(5,2)) as Pct_InsufficientService,
        SUM(Flag_NoQSI) as Count_NoQSI,
        CAST(100.0 * SUM(Flag_NoQSI) / NULLIF(@RecordCount, 0) AS DECIMAL(5,2)) as Pct_NoQSI
    FROM #ExclusionAnalysis
    
    -- =============================================
    -- RESULT SET 3: Consumer Level Analysis
    -- =============================================
    ;WITH ConsumerSummary AS (
        SELECT 
            CaseNo,
            COUNT(*) as YearsOfData,
            SUM(CASE WHEN Flag_MidYearQSI = 0 
                    AND Flag_LateEntry = 0 
                    AND Flag_EarlyExit = 0 
                    AND Flag_NoCosts = 0 
                    AND Flag_NegativeCosts = 0 
                    AND Flag_InsufficientService = 0 
                    AND Flag_NoQSI = 0 
                THEN 1 ELSE 0 END) as UsableYears,
            AVG(TotalPaidAmt) as AvgAnnualCost,
            MIN(FiscalYear) as FirstYear,
            MAX(FiscalYear) as LastYear
        FROM #ExclusionAnalysis
        GROUP BY CaseNo
    )
    SELECT 
        'Consumer Level Summary' as ReportSection,
        COUNT(*) as TotalConsumers,
        SUM(CASE WHEN UsableYears >= 1 THEN 1 ELSE 0 END) as ConsumersWithUsableData,
        CAST(100.0 * SUM(CASE WHEN UsableYears >= 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS DECIMAL(5,2)) as Pct_ConsumersUsable,
        SUM(CASE WHEN UsableYears >= 2 THEN 1 ELSE 0 END) as ConsumersWithMultiYear,
        CAST(100.0 * SUM(CASE WHEN UsableYears >= 2 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS DECIMAL(5,2)) as Pct_ConsumersMultiYear,
        AVG(CAST(YearsOfData AS FLOAT)) as AvgYearsPerConsumer,
        AVG(CAST(UsableYears AS FLOAT)) as AvgUsableYearsPerConsumer
    FROM ConsumerSummary
    
    -- =============================================
    -- RESULT SET 4: Fiscal Year Distribution
    -- =============================================
    SELECT 
        FiscalYear,
        COUNT(DISTINCT CaseNo) as ConsumersInYear,
        COUNT(*) as TotalRecords,
        SUM(CASE WHEN Flag_MidYearQSI = 0 
                AND Flag_LateEntry = 0 
                AND Flag_EarlyExit = 0 
                AND Flag_NoCosts = 0 
                AND Flag_NegativeCosts = 0 
                AND Flag_InsufficientService = 0 
                AND Flag_NoQSI = 0 
            THEN 1 ELSE 0 END) as UsableRecords,
        AVG(TotalPaidAmt) as AvgCost,
        STDEV(TotalPaidAmt) as StDevCost
    FROM #ExclusionAnalysis
    GROUP BY FiscalYear
    ORDER BY FiscalYear
    
    -- =============================================
    -- RESULT SET 5: Exclusion Overlap Analysis
    -- =============================================
    SELECT 
        'Exclusion Overlap' as ReportSection,
        SUM(CASE WHEN (Flag_MidYearQSI + Flag_LateEntry + Flag_EarlyExit + 
                      Flag_NoCosts + Flag_NegativeCosts + Flag_InsufficientService + 
                      Flag_NoQSI) = 0 THEN 1 ELSE 0 END) as NoExclusions,
        SUM(CASE WHEN (Flag_MidYearQSI + Flag_LateEntry + Flag_EarlyExit + 
                      Flag_NoCosts + Flag_NegativeCosts + Flag_InsufficientService + 
                      Flag_NoQSI) = 1 THEN 1 ELSE 0 END) as OneExclusion,
        SUM(CASE WHEN (Flag_MidYearQSI + Flag_LateEntry + Flag_EarlyExit + 
                      Flag_NoCosts + Flag_NegativeCosts + Flag_InsufficientService + 
                      Flag_NoQSI) = 2 THEN 1 ELSE 0 END) as TwoExclusions,
        SUM(CASE WHEN (Flag_MidYearQSI + Flag_LateEntry + Flag_EarlyExit + 
                      Flag_NoCosts + Flag_NegativeCosts + Flag_InsufficientService + 
                      Flag_NoQSI) >= 3 THEN 1 ELSE 0 END) as ThreeOrMoreExclusions
    FROM #ExclusionAnalysis
    
    -- =============================================
    -- RESULT SET 6: Cost Distribution for Included vs Excluded
    -- =============================================
    SELECT 
        CASE WHEN Flag_MidYearQSI = 0 
                AND Flag_LateEntry = 0 
                AND Flag_EarlyExit = 0 
                AND Flag_NoCosts = 0 
                AND Flag_NegativeCosts = 0 
                AND Flag_InsufficientService = 0 
                AND Flag_NoQSI = 0 
            THEN 'Included' 
            ELSE 'Excluded' 
        END as DataStatus,
        COUNT(*) as RecordCount,
        AVG(TotalPaidAmt) as AvgCost,
        STDEV(TotalPaidAmt) as StDevCost,
        MIN(TotalPaidAmt) as MinCost,
        MAX(TotalPaidAmt) as MaxCost
    FROM #ExclusionAnalysis
    GROUP BY 
        CASE WHEN Flag_MidYearQSI = 0 
                AND Flag_LateEntry = 0 
                AND Flag_EarlyExit = 0 
                AND Flag_NoCosts = 0 
                AND Flag_NegativeCosts = 0 
                AND Flag_InsufficientService = 0 
                AND Flag_NoQSI = 0 
            THEN 'Included' 
            ELSE 'Excluded' 
        END
    
    -- =============================================
    -- RESULT SET 7: Detailed Records for Python Analysis  
    -- =============================================
    SELECT 
        CaseNo,
        FiscalYear,
        TotalPaidAmt,
        ServiceDays,
        DaysFromFYStart,
        DaysToFYEnd,
        Flag_MidYearQSI,
        Flag_LateEntry,
        Flag_EarlyExit,
        Flag_NoCosts,
        Flag_NegativeCosts,
        Flag_InsufficientService,
        Flag_NoQSI,
        CASE WHEN Flag_MidYearQSI = 0 
                AND Flag_LateEntry = 0 
                AND Flag_EarlyExit = 0 
                AND Flag_NoCosts = 0 
                AND Flag_NegativeCosts = 0 
                AND Flag_InsufficientService = 0 
                AND Flag_NoQSI = 0 
            THEN 1 ELSE 0 END as IsUsable
    FROM #ExclusionAnalysis
    ORDER BY CaseNo, FiscalYear
    
    -- =============================================
    -- Clean up
    -- =============================================
    IF OBJECT_ID('tempdb..#FiscalYearClaims') IS NOT NULL
        DROP TABLE #FiscalYearClaims
    
    IF OBJECT_ID('tempdb..#ConsumerYearSummary') IS NOT NULL
        DROP TABLE #ConsumerYearSummary
    
    IF OBJECT_ID('tempdb..#MidYearQSI') IS NOT NULL
        DROP TABLE #MidYearQSI
    
    IF OBJECT_ID('tempdb..#ExclusionAnalysis') IS NOT NULL
        DROP TABLE #ExclusionAnalysis
    
END