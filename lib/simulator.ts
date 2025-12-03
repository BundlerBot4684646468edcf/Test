// Core simulation logic for Hotel Decision Simulator

export type SimulationInput = {
  requestsPerDay: number;
  numReceptionists: number;
  salary: number;
  aiCost: number;
  aiCanHandle: number; // 0–1
  reqPerHourHuman?: number;
  reqPerHourAI?: number;
  workingHours?: number;
  humanUpsellRate: number; // 0–1
  aiUpsellRate: number; // 0–1
  avgUpsellValue: number;
};

export type SimulationOutput = {
  humanCapacity: number;
  aiCapacity: number;
  totalCapacity: number;
  loadFactor: number;
  aiHandled: number;
  humanHandled: number;
  unhandled: number;
  satisfaction: number;
  upsellRevenueMonth: number;
  staffCost: number;
  totalCost: number;
  netGain: number;
};

/**
 * Simulates hotel reception staffing scenarios
 * Calculates capacity, satisfaction, costs, and net gain
 */
export function simulate(input: SimulationInput): SimulationOutput {
  const {
    requestsPerDay,
    numReceptionists,
    salary,
    aiCost,
    aiCanHandle,
    reqPerHourHuman = 15,
    reqPerHourAI = 40,
    workingHours = 8,
    humanUpsellRate,
    aiUpsellRate,
    avgUpsellValue
  } = input;

  // Calculate daily capacity for humans and AI
  const humanCapacity = numReceptionists * reqPerHourHuman * workingHours;
  const aiCapacity = aiCost > 0 ? reqPerHourAI * workingHours : 0;
  const totalCapacity = humanCapacity + aiCapacity;

  // Load factor: ratio of demand to capacity
  const loadFactor = totalCapacity > 0 ? requestsPerDay / totalCapacity : Infinity;

  // Distribution of requests handled
  const aiHandled = Math.min(requestsPerDay * aiCanHandle, aiCapacity);
  const humanHandled = Math.min(requestsPerDay - aiHandled, humanCapacity);
  const unhandled = Math.max(0, requestsPerDay - aiHandled - humanCapacity);

  // Guest satisfaction calculation (0-100)
  let satisfaction = 80;
  if (loadFactor > 1) satisfaction -= 15; // Overloaded staff reduces satisfaction
  if (unhandled > 0) satisfaction -= 10; // Unhandled requests reduce satisfaction
  satisfaction += aiCanHandle * 10; // AI can improve satisfaction with instant responses
  satisfaction = Math.min(100, Math.max(0, satisfaction));

  // Upsell revenue calculation
  const humanUpsells = humanHandled * humanUpsellRate;
  const aiUpsells = aiHandled * aiUpsellRate;
  const upsellRevenueDay = (humanUpsells + aiUpsells) * avgUpsellValue;
  const upsellRevenueMonth = upsellRevenueDay * 30;

  // Cost calculation
  const staffCost = numReceptionists * salary;
  const totalCost = staffCost + aiCost;

  // Net gain/loss
  const netGain = upsellRevenueMonth - totalCost;

  return {
    humanCapacity,
    aiCapacity,
    totalCapacity,
    loadFactor,
    aiHandled,
    humanHandled,
    unhandled,
    satisfaction,
    upsellRevenueMonth,
    staffCost,
    totalCost,
    netGain
  };
}
