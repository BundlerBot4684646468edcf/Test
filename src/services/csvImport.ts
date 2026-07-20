import { parse } from 'csv-parse/sync';
import { customers } from '../db';

export interface CustomerImportRow {
  firstName: string;
  phone?: string;
  email?: string;
  servedAt: string;
  servedBy?: string;
}

export async function importCustomersFromCSV(
  businessId: string,
  csvContent: string,
  source: 'past' | 'new' = 'past'
): Promise<{
  imported: number;
  errors: Array<{ row: number; error: string }>;
}> {
  const errors: Array<{ row: number; error: string }> = [];
  let imported = 0;

  let records: CustomerImportRow[];
  try {
    records = parse(csvContent, {
      columns: true,
      skip_empty_lines: true,
      trim: true,
    }) as CustomerImportRow[];
  } catch (error) {
    throw new Error(`CSV parsing failed: ${String(error)}`);
  }

  for (let i = 0; i < records.length; i++) {
    const row = records[i];
    try {
      const { firstName, phone, email, servedAt, servedBy } = row;

      if (!firstName || !servedAt) {
        errors.push({ row: i + 2, error: 'firstName and servedAt are required' });
        continue;
      }
      if (!phone && !email) {
        errors.push({ row: i + 2, error: 'At least phone or email is required' });
        continue;
      }

      const parsedDate = new Date(servedAt);
      if (isNaN(parsedDate.getTime())) {
        errors.push({ row: i + 2, error: 'Invalid date format for servedAt' });
        continue;
      }

      customers.create({
        businessId,
        firstName,
        phone: phone || null,
        email: email || null,
        servedAt: parsedDate.toISOString(),
        servedBy: servedBy || null,
        source,
      });

      imported++;
    } catch (error) {
      errors.push({ row: i + 2, error: String(error) });
    }
  }

  return { imported, errors };
}
