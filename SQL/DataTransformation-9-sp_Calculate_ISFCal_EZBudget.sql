USE [APD]
GO

-- Drop procedure if it exists
IF EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[sp_Calculate_ISFCal_EZBudget]') AND type in (N'P', N'PC'))
DROP PROCEDURE [dbo].[sp_Calculate_ISFCal_EZBudget]
GO

CREATE PROCEDURE [dbo].[sp_Calculate_ISFCal_EZBudget]
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
        EZBudgetAssessId bigint,
        CASENO bigint,
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
    INSERT INTO #ISFCalculations
    SELECT 
        e.EZBudgetAssessId,
        e.CASENO,
        -- Base value for all individuals
        27.5720 AS BaseValue,
        
        -- Age weight
        CASE 
            WHEN ISNUMERIC(e.CurrentAge) = 1 THEN
                CASE 
                    WHEN CAST(e.CurrentAge AS int) BETWEEN 21 AND 30 THEN 47.8473
                    WHEN CAST(e.CurrentAge AS int) >= 31 THEN 48.9634
                    ELSE 0
                END
            ELSE 0
        END AS AgeWeight,
        
        -- Living Setting weight
        CASE 
            WHEN e.LivingSetting LIKE '%Independent Living%' 
                OR e.LivingSetting LIKE '%Supported Living%' THEN 35.8220
            WHEN e.LivingSetting LIKE '%Standard%Live-In%' THEN 90.6294
            WHEN e.LivingSetting LIKE '%Behavior Focus%' THEN 131.7576
            WHEN e.LivingSetting LIKE '%Intensive Behavior%' THEN 209.4558
            WHEN e.LivingSetting LIKE '%CTEP%' 
                OR e.LivingSetting LIKE '%Special Medical Home Care%' THEN 267.0995
            ELSE 0 -- Family Home (base case)
        END AS LivingSettingWeight,
        
        -- Behavioral Sum (Q25-Q30) * 0.4954
        (ISNULL(TRY_CAST(e.Q25 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q26 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q27 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q28 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q29 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q30 AS float), 0)) * 0.4954 AS BehavioralSum,
        
        -- Functional Sum (Q14-Q24)
        (ISNULL(TRY_CAST(e.Q14 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q15 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q16 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q17 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q18 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q19 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q20 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q21 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q22 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q23 AS float), 0) + 
         ISNULL(TRY_CAST(e.Q24 AS float), 0)) AS FunctionalSum,
        
        -- Interaction: Family Home * Functional Sum * 0.6349
        CASE 
            WHEN e.LivingSetting LIKE '%Family Home%' THEN
                (ISNULL(TRY_CAST(e.Q14 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q15 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q16 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q17 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q18 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q19 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q20 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q21 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q22 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q23 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q24 AS float), 0)) * 0.6349
            ELSE 0
        END AS InteractionFHF,
        
        -- Interaction: ILSL * Functional Sum * 2.0529
        CASE 
            WHEN e.LivingSetting LIKE '%Independent Living%' 
                OR e.LivingSetting LIKE '%Supported Living%' THEN
                (ISNULL(TRY_CAST(e.Q14 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q15 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q16 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q17 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q18 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q19 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q20 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q21 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q22 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q23 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q24 AS float), 0)) * 2.0529
            ELSE 0
        END AS InteractionSLF,
        
        -- Interaction: ILSL * Behavioral Sum * 1.4501
        CASE 
            WHEN e.LivingSetting LIKE '%Independent Living%' 
                OR e.LivingSetting LIKE '%Supported Living%' THEN
                (ISNULL(TRY_CAST(e.Q25 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q26 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q27 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q28 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q29 AS float), 0) + 
                 ISNULL(TRY_CAST(e.Q30 AS float), 0)) * 1.4501
            ELSE 0
        END AS InteractionSLB,
        
        -- Individual question weights
        ISNULL(TRY_CAST(e.Q16 AS float), 0) * 2.4984 AS Q16Weight,
        ISNULL(TRY_CAST(e.Q18 AS float), 0) * 5.8537 AS Q18Weight,
        ISNULL(TRY_CAST(e.Q20 AS float), 0) * 2.6772 AS Q20Weight,
        ISNULL(TRY_CAST(e.Q21 AS float), 0) * 2.7878 AS Q21Weight,
        ISNULL(TRY_CAST(e.Q23 AS float), 0) * 6.3555 AS Q23Weight,
        ISNULL(TRY_CAST(e.Q28 AS float), 0) * 2.2803 AS Q28Weight,
        ISNULL(TRY_CAST(e.Q33 AS float), 0) * 1.2233 AS Q33Weight,
        ISNULL(TRY_CAST(e.Q34 AS float), 0) * 2.1764 AS Q34Weight,
        ISNULL(TRY_CAST(e.Q36 AS float), 0) * 2.6734 AS Q36Weight,
        ISNULL(TRY_CAST(e.Q43 AS float), 0) * 1.9304 AS Q43Weight,
        
        0 AS TotalSum,  -- Will be calculated next
        0 AS ISFCal     -- Will be calculated next
        
    FROM [dbo].[tbl_EZBudget] e
    WHERE (@CaseNo IS NULL OR e.CASENO = @CaseNo);
    
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
        SELECT TOP 10
            EZBudgetAssessId,
            CASENO,
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
        ORDER BY EZBudgetAssessId;
    END
    
    -- Update the main table with calculated ISFCal values
    UPDATE e
    SET e.ISFCal = c.ISFCal
    FROM [dbo].[tbl_EZBudget] e
    INNER JOIN #ISFCalculations c ON e.EZBudgetAssessId = c.EZBudgetAssessId;
    
    SET @RowsUpdated = @@ROWCOUNT;
    
    -- Clean up
    DROP TABLE #ISFCalculations;
    
    -- Return summary
    SELECT 
        @RowsUpdated AS RowsUpdated,
        DATEDIFF(SECOND, @StartTime, GETDATE()) AS ExecutionTimeSeconds,
        'ISFCal calculation completed for tbl_EZBudget' AS Status;
    
END
GO

-- Grant execute permissions
GRANT EXECUTE ON [dbo].[sp_Calculate_ISFCal_EZBudget] TO [public]
GO