USE [APD]
GO

-- Drop procedure if it exists
IF EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[sp_Calculate_ISFCal_QSIAssessments]') AND type in (N'P', N'PC'))
DROP PROCEDURE [dbo].[sp_Calculate_ISFCal_QSIAssessments]
GO

CREATE PROCEDURE [dbo].[sp_Calculate_ISFCal_QSIAssessments]
    @CaseNo bigint = NULL,  -- Optional: Calculate for specific case only
    @Debug bit = 0           -- Optional: Show debug information
AS
BEGIN
    SET NOCOUNT ON;
    
    -- Declare variables for tracking
    DECLARE @RowsUpdated int = 0;
    DECLARE @StartTime datetime = GETDATE();
    
    -- Create temp table for calculations
    CREATE TABLE #ISFCalculations (
        AssessID bigint,
        CASENO bigint,
        ResidenceType varchar(200),
        CurrentAge int,
        BaseValue float,
        AgeWeight float,
        LivingSettingWeight float,
        BehavioralSum float,
        FunctionalSum float,
        InteractionFHF float,
        InteractionSLF float,
        InteractionSLB float,
        Q16Weight float,
        Q18Weight float,
        Q20Weight float,
        Q21Weight float,
        Q23Weight float,
        Q28Weight float,
        Q33Weight float,
        Q34Weight float,
        Q36Weight float,
        Q43Weight float,
        TotalSum float,
        ISFCal float
    );
    
    -- Insert calculations into temp table
    -- Join with tbl_Consumers to get RESIDENCETYPE and age
    INSERT INTO #ISFCalculations
    SELECT 
        q.AssessID,
        q.CASENO,
        c.RESIDENCETYPE,
        DATEDIFF(YEAR, c.DOB, GETDATE()) AS CurrentAge,
        
        -- Base value for all individuals
        27.5720 AS BaseValue,
        
        -- Age weight
        CASE 
            WHEN DATEDIFF(YEAR, c.DOB, GETDATE()) BETWEEN 21 AND 30 THEN 47.8473
            WHEN DATEDIFF(YEAR, c.DOB, GETDATE()) >= 31 THEN 48.9634
            ELSE 0
        END AS AgeWeight,
        
        -- Living Setting weight based on RESIDENCETYPE mapping
        CASE 
            -- ILSL: Independent Living, Supported Living, or facilities without RH services
            WHEN c.RESIDENCETYPE IN ('Independent Living', 'Supported Living') THEN 35.8220
            WHEN c.RESIDENCETYPE LIKE '%Nursing Home%' 
                AND NOT EXISTS (
                    SELECT 1 FROM [dbo].[tbl_PlannedServices] ps 
                    WHERE ps.CaseNo = q.CASENO 
                    AND ps.Service LIKE '%Residential Habilitation%'
                ) THEN 35.8220
                
            -- RH1: Standard Residential Habilitation or Live-In
            -- Most APD Licensed Facilities fall into this category by default
            WHEN c.RESIDENCETYPE LIKE 'APD Licensed Facility%Small Group Home%' THEN 90.6294
            WHEN c.RESIDENCETYPE LIKE 'APD Licensed Facility%Large Group Home%' THEN 90.6294
            WHEN c.RESIDENCETYPE LIKE 'APD Licensed Facility%Foster Home%' THEN 90.6294
            WHEN c.RESIDENCETYPE LIKE 'DCF Licensed Home%Small Group Home%' THEN 90.6294
            WHEN c.RESIDENCETYPE LIKE 'DCF Licensed Home%Large Group Home%' THEN 90.6294
            WHEN c.RESIDENCETYPE LIKE 'DCF Licensed Home%Foster Home%' THEN 90.6294
            WHEN c.RESIDENCETYPE = 'DCF Licensed Home' THEN 90.6294
            WHEN c.RESIDENCETYPE LIKE 'AHCA Licensed Assisted Living%' THEN 90.6294
            WHEN c.RESIDENCETYPE LIKE 'AHCA Licensed Adult Family Care%' THEN 90.6294
            WHEN c.RESIDENCETYPE LIKE 'AHCA Licensed Private ICF/DD%' THEN 90.6294
            
            -- RH2: Behavior Focus (would need additional data to identify)
            -- This would typically be identified by facility designation
            -- For now, using Residential Habilitation Center as proxy
            WHEN c.RESIDENCETYPE LIKE '%Residential Habilitation Center%' THEN 131.7576
            
            -- RH3: Intensive Behavior
            -- APD Developmental Disabilities Centers and Defendant Programs
            WHEN c.RESIDENCETYPE LIKE 'APD Developmental Disabilities Center%' THEN 209.4558
            WHEN c.RESIDENCETYPE LIKE 'APD Developmental Disabilities Defendant%' THEN 209.4558
            WHEN c.RESIDENCETYPE LIKE '%Psychiatric%' THEN 209.4558
            WHEN c.RESIDENCETYPE LIKE 'Mental Health Placement%' THEN 209.4558
            
            -- RH4: CTEP or Special Medical Home Care
            -- This would need to be identified by specific program enrollment
            -- For now, leaving as placeholder
            
            -- Family Home (base case)
            WHEN c.RESIDENCETYPE = 'Family Home' THEN 0
            
            -- Default to 0 for unmapped types
            ELSE 0
        END AS LivingSettingWeight,
        
        -- Behavioral Sum (Q25-Q30) * 0.4954
        (ISNULL(TRY_CAST(q.Q25 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q26 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q27 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q28 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q29 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q30 AS float), 0)) * 0.4954 AS BehavioralSum,
        
        -- Functional Sum (Q14-Q24)
        (ISNULL(TRY_CAST(q.Q14 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q15 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q16 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q17 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q18 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q19 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q20 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q21 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q22 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q23 AS float), 0) + 
         ISNULL(TRY_CAST(q.Q24 AS float), 0)) AS FunctionalSum,
        
        -- Interaction: Family Home * Functional Sum * 0.6349
        CASE 
            WHEN c.RESIDENCETYPE = 'Family Home' THEN
                (ISNULL(TRY_CAST(q.Q14 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q15 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q16 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q17 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q18 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q19 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q20 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q21 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q22 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q23 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q24 AS float), 0)) * 0.6349
            ELSE 0
        END AS InteractionFHF,
        
        -- Interaction: ILSL * Functional Sum * 2.0529
        CASE 
            WHEN c.RESIDENCETYPE IN ('Independent Living', 'Supported Living') THEN
                (ISNULL(TRY_CAST(q.Q14 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q15 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q16 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q17 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q18 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q19 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q20 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q21 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q22 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q23 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q24 AS float), 0)) * 2.0529
            ELSE 0
        END AS InteractionSLF,
        
        -- Interaction: ILSL * Behavioral Sum * 1.4501
        CASE 
            WHEN c.RESIDENCETYPE IN ('Independent Living', 'Supported Living') THEN
                (ISNULL(TRY_CAST(q.Q25 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q26 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q27 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q28 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q29 AS float), 0) + 
                 ISNULL(TRY_CAST(q.Q30 AS float), 0)) * 1.4501
            ELSE 0
        END AS InteractionSLB,
        
        -- Individual question weights
        ISNULL(TRY_CAST(q.Q16 AS float), 0) * 2.4984 AS Q16Weight,
        ISNULL(TRY_CAST(q.Q18 AS float), 0) * 5.8537 AS Q18Weight,
        ISNULL(TRY_CAST(q.Q20 AS float), 0) * 2.6772 AS Q20Weight,
        ISNULL(TRY_CAST(q.Q21 AS float), 0) * 2.7878 AS Q21Weight,
        ISNULL(TRY_CAST(q.Q23 AS float), 0) * 6.3555 AS Q23Weight,
        ISNULL(TRY_CAST(q.Q28 AS float), 0) * 2.2803 AS Q28Weight,
        ISNULL(TRY_CAST(q.Q33 AS float), 0) * 1.2233 AS Q33Weight,
        ISNULL(TRY_CAST(q.Q34 AS float), 0) * 2.1764 AS Q34Weight,
        ISNULL(TRY_CAST(q.Q36 AS float), 0) * 2.6734 AS Q36Weight,
        ISNULL(TRY_CAST(q.Q43 AS float), 0) * 1.9304 AS Q43Weight,
        
        0 AS TotalSum,  -- Will be calculated next
        0 AS ISFCal     -- Will be calculated next
        
    FROM [dbo].[tbl_QSIAssessments] q
    INNER JOIN [dbo].[tbl_Consumers] c ON q.CASENO = c.CASENO
    WHERE (@CaseNo IS NULL OR q.CASENO = @CaseNo);
    
    -- Calculate TotalSum (sum of all weighted values)
    UPDATE #ISFCalculations
    SET TotalSum = BaseValue + AgeWeight + LivingSettingWeight + BehavioralSum + 
                   InteractionFHF + InteractionSLF + InteractionSLB +
                   Q16Weight + Q18Weight + Q20Weight + Q21Weight + Q23Weight + 
                   Q28Weight + Q33Weight + Q34Weight + Q36Weight + Q43Weight;
    
    -- Calculate ISFCal (square of TotalSum)
    UPDATE #ISFCalculations
    SET ISFCal = POWER(TotalSum, 2);
    
    -- Show debug information if requested
    IF @Debug = 1
    BEGIN
        -- Show calculation breakdown for first 10 records
        SELECT TOP 10
            AssessID,
            CASENO,
            ResidenceType,
            CurrentAge,
            BaseValue,
            AgeWeight,
            LivingSettingWeight,
            BehavioralSum,
            InteractionFHF,
            InteractionSLF,
            InteractionSLB,
            TotalSum,
            ISFCal
        FROM #ISFCalculations
        ORDER BY AssessID;
        
        -- Show residence type mapping summary
        SELECT 
            ResidenceType,
            COUNT(*) AS Count,
            AVG(LivingSettingWeight) AS AvgLivingSettingWeight,
            MIN(ISFCal) AS MinISFCal,
            AVG(ISFCal) AS AvgISFCal,
            MAX(ISFCal) AS MaxISFCal
        FROM #ISFCalculations
        GROUP BY ResidenceType
        ORDER BY COUNT(*) DESC;
    END
    
    -- Update the main table with calculated ISFCal values
    UPDATE q
    SET q.ISFCal = c.ISFCal
    FROM [dbo].[tbl_QSIAssessments] q
    INNER JOIN #ISFCalculations c ON q.AssessID = c.AssessID;
    
    SET @RowsUpdated = @@ROWCOUNT;
    
    -- Clean up
    DROP TABLE #ISFCalculations;
    
    -- Return summary
    SELECT 
        @RowsUpdated AS RowsUpdated,
        DATEDIFF(SECOND, @StartTime, GETDATE()) AS ExecutionTimeSeconds,
        'ISFCal calculation completed for tbl_QSIAssessments' AS Status;
    
END
GO

-- Grant execute permissions
GRANT EXECUTE ON [dbo].[sp_Calculate_ISFCal_QSIAssessments] TO [public]
GO

/*
USAGE EXAMPLES:

-- Calculate ISFCal for all records in tbl_QSIAssessments
EXEC sp_Calculate_ISFCal_QSIAssessments;

-- Calculate ISFCal for a specific CASENO with debug output
EXEC sp_Calculate_ISFCal_QSIAssessments @CaseNo = 12345, @Debug = 1;

-- View results after calculation
SELECT TOP 100 
    AssessID, 
    CASENO, 
    ISFCal 
FROM tbl_QSIAssessments 
WHERE ISFCal IS NOT NULL
ORDER BY ISFCal DESC;

NOTES ON RESIDENCE TYPE MAPPING:

This procedure maps RESIDENCETYPE values from tbl_Consumers to Model 5b living setting categories.
The mapping may need refinement based on:

1. Actual facility designations (Behavior Focus vs Standard)
2. Specific program enrollments (CTEP, Special Medical Home Care)
3. Whether facilities provide Residential Habilitation services

Consider creating a lookup table for more accurate mapping:
CREATE TABLE tbl_ResidenceTypeMapping (
    RESIDENCETYPE varchar(200),
    Model5bCategory varchar(50),
    LivingSettingWeight float
);
*/