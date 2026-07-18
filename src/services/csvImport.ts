import { parse } from 'csv-parse/sync';
import prisma from '../db';

export interface CustomerImportRow {
  firstName: string;
  phone?: string;
  email?: string;
  servedAt: string;
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

  try {
    const records = parse(csvContent, {
      columns: true,
      skip_empty_lines: true,
      trim: true,
    }) as CustomerImportRow[];

    for (let i = 0; i < records.length; i++) {
      const row = records[i];
      try {
        const { firstName, phone, email, servedAt } = row;

        if (!firstName || !servedAt) {
          errors.push({
            row: i + 2,
            error: 'firstName and servedAt are required',
          });
          continue;
        }

        // Validate at least one contact method
        if (!phone && !email) {
          errors.push({
            row: i + 2,
            error: 'At least phone or email is required',
          });
          continue;
        }

        const parsedDate = new Date(servedAt);
        if (isNaN(parsedDate.getTime())) {
          errors.push({
            row: i + 2,
            error: 'Invalid date format for servedAt',
          });
          continue;
        }

        await prisma.customer.create({
          data: {
            businessId,
            firstName,
            phone: phone || null,
            email: email || null,
            servedAt: parsedDate,
            source,
          },
        });

        imported++;
      } catch (error) {
        errors.push({
          row: i + 2,
          error: String(error),
        });
      }
    }
  } catch (error) {
    throw new Error(`CSV parsing failed: ${String(error)}`);
  }

  return { imported, errors };
}
