import express from 'express';
import multer from 'multer';
import { customers } from '../db';
import { importCustomersFromCSV } from '../services/csvImport';

const router = express.Router({ mergeParams: true });
const upload = multer({ storage: multer.memoryStorage() });

// GET /api/businesses/:businessId/customers — List customers
router.get('/', async (req, res) => {
  try {
    const { businessId } = req.params as Record<string, string>;
    const skip = parseInt((req.query.skip as string) || '0', 10);
    const take = parseInt((req.query.take as string) || '20', 10);

    const list = customers.list(businessId, skip, take);
    const total = customers.count(businessId);

    res.json({ customers: list, total });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to list customers' });
  }
});

// POST /api/businesses/:businessId/customers/import — Import from CSV
router.post('/import', upload.single('file'), async (req, res) => {
  try {
    const { businessId } = req.params as Record<string, string>;
    const source = req.body.source === 'new' ? 'new' : 'past';
    const consented = req.body.consent === 'true' || req.body.consent === true;

    if (!req.file) {
      return res.status(400).json({ error: 'No file provided' });
    }
    if (!consented) {
      return res.status(400).json({
        error: 'Consent confirmation required before importing customers',
      });
    }

    const csvContent = req.file.buffer.toString('utf-8');
    const result = await importCustomersFromCSV(businessId, csvContent, source, consented);

    res.json({
      success: true,
      imported: result.imported,
      errors: result.errors,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: String(error) });
  }
});

// DELETE /api/businesses/:businessId/customers/:customerId
router.delete('/:customerId', async (req, res) => {
  try {
    const { businessId, customerId } = req.params as Record<string, string>;

    const customer = customers.get(customerId);
    if (!customer || customer.businessId !== businessId) {
      return res.status(404).json({ error: 'Customer not found' });
    }

    customers.delete(customerId);
    res.json({ success: true });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to delete customer' });
  }
});

export default router;
