-- SQL Server Script to Copy specific tables from APDFees database to dbo schema
-- This will copy tbl_Rates and tbl_ServiceCodes to dbo schema for easier querying

-- Drop existing tables in dbo schema if they exist
DROP TABLE IF EXISTS [dbo].[tbl_Rates];
DROP TABLE IF EXISTS [dbo].[tbl_ServiceCodes];

-- Copy tbl_Rates from APDFees database to dbo schema in current database
SELECT * 
INTO [dbo].[tbl_Rates]
FROM [APDFees].[dbo].[tbl_Rates];

PRINT 'Copied table: tbl_Rates from APDFees.dbo to dbo';
PRINT 'Rows copied: ' + CAST(@@ROWCOUNT AS VARCHAR(10));
PRINT '';

-- Copy tbl_ServiceCodes from APDFees database to dbo schema in current database
SELECT * 
INTO [dbo].[tbl_ServiceCodes]
FROM [APDFees].[dbo].[tbl_ServiceCodes];

PRINT 'Copied table: tbl_ServiceCodes from APDFees.dbo to dbo';
PRINT 'Rows copied: ' + CAST(@@ROWCOUNT AS VARCHAR(10));
PRINT '';

-- Verify the tables were created successfully
PRINT 'Verification - You can now query these tables without schema prefix:';
PRINT 'SELECT * FROM tbl_Rates';
PRINT 'SELECT * FROM tbl_ServiceCodes';
PRINT '';

-- Show row counts for the new tables
PRINT 'Row counts for copied tables:';
SELECT 'tbl_Rates' AS TableName, COUNT(*) AS [RowCount] FROM [dbo].[tbl_Rates]
UNION ALL
SELECT 'tbl_ServiceCodes' AS TableName, COUNT(*) AS [RowCount] FROM [dbo].[tbl_ServiceCodes];

-- Clean up the APD schema tables if they were already created
IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'APD' AND TABLE_NAME = 'tbl_Rates')
BEGIN
    DROP TABLE [APD].[tbl_Rates];
    PRINT '';
    PRINT 'Cleaned up: Removed APD.tbl_Rates';
END

IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'APD' AND TABLE_NAME = 'tbl_ServiceCodes')
BEGIN
    DROP TABLE [APD].[tbl_ServiceCodes];
    PRINT 'Cleaned up: Removed APD.tbl_ServiceCodes';
END