# TGF MVP Hardware Guide

Hardware specifications for deploying TGF on a real cooling tower.

## MVP Sensors (4 Essential)

| Sensor | Purpose | Range | Accuracy | Cost |
|--------|---------|-------|----------|------|
| pH | Scale/corrosion control | 0-14 | +/-0.01 | $800 |
| Conductivity | CoC monitoring, blowdown control | 0-10,000 uS/cm | +/-2% | $700 |
| Temperature | Reaction rates, sensor compensation | 0-50C | +/-0.1C | $300 |
| ORP | Disinfectant efficacy | 0-1000 mV | +/-5mV | $900 |

**Total sensor cost**: $2,700

## MVP Actuators (4 Essential)

| Actuator | Chemical | Flow Range | Control | Cost |
|----------|----------|------------|---------|------|
| Pump 1 | Scale & Corrosion Inhibitor | 0.1-10 mL/min | 4-20mA | $1,000 |
| Pump 2 | Oxidizing Biocide | 0.1-5 mL/min | 4-20mA | $1,000 |
| Pump 3 | Non-Oxidizing Biocide | 0.1-5 mL/min | 4-20mA | $1,000 |
| Valve | Automated Blowdown | 0-100% open | Modbus | $800 |

**Total actuator cost**: $3,800

## Edge Device

| Component | Cost |
|-----------|------|
| Raspberry Pi 4 (4GB) | $75 |
| 4G USB Modem | $60 |
| Industrial enclosure (IP54) | $150 |
| MicroSD 64GB | $15 |
| Power supply + UPS (2h) | $200 |
| Modbus RTU/RS485 converter | $50 |
| **Total** | **$550** |

## Total Per-Tower Cost

| Item | Cost |
|------|------|
| Sensors | $2,700 |
| Actuators | $3,800 |
| Edge device | $550 |
| Installation | $3,500 |
| Training (4h) | $500 |
| Contingency (10%) | $950 |
| **Total** | **$12,000** |

## ROI (Annual, Per Tower)

| Item | Amount |
|------|--------|
| Chemical savings (15%) | $4,500 |
| Water savings (10%) | $2,000 |
| Labor savings | $5,000 |
| **Total savings** | **$11,500/year** |
| Operating cost | -$2,040/year |
| **Net benefit** | **$9,460/year** |
| **Payback period** | **15 months** |

## MVP Success Criteria (30-Day Pilot)

| Metric | Target |
|--------|--------|
| Uptime | >95% |
| Anomaly detection F1 | >0.80 |
| False positive rate | <10% |
| pH stability | +/-0.3 |
| Human intervention | 0 times |
| Chemical reduction | >15% |
