# Golden eval set — manual review

100 items, 85 awaiting verification.

For each item check three things: the question sounds like a real
customer, the answer is correct and complete, and the page is right.
Then edit `golden.jsonl` — fix the text if needed and set
`"verified": true`. `run_ragas.py` skips unverified items.

---

## `g-001` · lookup · starhealth starhealth__health__comprehensive.pdf · p.38

**Q:** What phone number can I call for help with my claim 24 hours a day?

**A:** 044-69006900 or Toll Free No. 1800 425 2255

*verified: False · vocab overlap: 0.429*

<details><summary>source chunk</summary>

```
f.
Receipts from doctors, surgeons,
anesthetist in original
g.
Certificate from the attending doctor
regarding the diagnosis.
h.
Copy of PAN card
i.
Copy of Aadhaar Card
j.
Any other document specific to the
treatment / illness
k.
Prescriptions and receipt for Pre and
Post-Hospitalization expenses
l.
KYC (Identity proof with Address) of
the proposer, as per AML Guidelines
m. NEFT
documents
viz.,
Customer
name, Bank Account No., Name of
the Bank, IFSC code
n.
CKYC No. of the proposer (if available)
Note: For assistance call 24 hours helpline 044-69006900 or Toll Free No. 1800
425 2255, Senior Citizens may call at
044-40020888
For
the
comprehensive
list
of
documents to be submitted while filing
a reimbursement claim, please refer
our website under the link https://www.
starhealth.in/claims/#claim-process
F.
Claims of Out Patient Consultations /
treatments will be settled on cashless
basis.
G.
For Accidental Death Claims: Claim Form
a.
Death Certificate
b.
```

</details>

---

## `g-002` · lookup · starhealth starhealth__health__comprehensive.pdf · p.42

**Q:** How much notice will I get if the insurance company decides to change the terms of my policy, including the price I pay?

**A:** The company will notify me thirty days before the changes are effected.

*verified: False · vocab overlap: 0.5*

<details><summary>source chunk</summary>

```
stipulated Grace Period.
iv. No interest will be charged if the
instalment premium is not paid on
due date.
v.
ln case of instalment premium due
not received within the Grace Period,
the policy will get cancelled.
vi. ln
the
event
of
a
claim,
all
subsequent premium instalments
shall immediately become due and
payable.
vii. The company has the right to
recover and deduct all the pending
instalments from the claim amount
due under the policy.
viii. For premium paid in instalments
during the Policy Period, coverage
is available during the Grace Period
also.
13. Possibility of Revision of Terms of the
Policy including the Premium Rates: The
Company may revise or modify the terms
of the policy including the premium rates
as per the extant Guidelines. The Insured
Person shall be notified thirty days before
the changes are effected.
14. Free Look Period: The Free Look Period
shall be applicable on new individual
health insurance policies and not on
renewals or at the time of porting/
```

</details>

---

## `g-003` · lookup · starhealth starhealth__health__comprehensive.pdf · p.16

**Q:** How often can I get a free health checkup per year?

**A:** once per Policy Year

*verified: False · vocab overlap: 0.571*

<details><summary>source chunk</summary>

```
arrange for a Preventive Health Checkup at Our Network Providers for the
applicable package as specified below
as per opted Sum Insured and subject to
the conditions specified below:
Sum Insured (Rs.)
Package applicable
5,00,000/- to
10,00,000/-
Package A
Above 10,00,000/-
Package B
i.
An initial waiting period of 30 days
shall apply from the first inception of
Policy. This waiting period shall not
be applicable during subsequent
renewals.
ii.
Health Check-up can be availed
once per Policy Year per Insured
Person who is covered as Adult in the
Policy and all the tests must have
been done on the same date.
```

</details>

---

## `g-004` · lookup · starhealth starhealth__health__comprehensive.pdf · p.20

**Q:** How much is the insurance payout if I lose my thumb?

**A:** 25

*verified: False · vocab overlap: 0.2*

<details><summary>source chunk</summary>

```
For each toe
1
2
Loss of hearing both ears
Both ears
75
Loss of hearing one ear
One ear
30
3
Loss of four fingers and
thumbs of One hand
40
4
Loss of four fingers
35
Loss of thumb both phalanges
Both phalanges
25
One phalanx
10
5
Loss of index finger three
phalanges
Three phalanges
10
Two phalanges
Two phalanges
8
One phalanx
One phalanx
4
```

</details>

---

## `g-005` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.12

**Q:** Can I add my new spouse or baby to the policy and still get some benefits right away?

**A:** Yes, you can get continuity benefit for waiting periods already served, including the first 30 days, specific illness, and pre-existing disease waiting periods.

*verified: False · vocab overlap: 0.273*

<details><summary>source chunk</summary>

```
ix. In a given Policy Year, either Endless Sum Insured (if opted)
or Restore Benefit (if opted) can be utilised.

x. Benefit 10.1 Voluntary Deductible or benefit 10.2
Voluntary Co-Payment, if opted shall be applicable.

xi. This benefit shall not be applicable to optional covers
3.1(h), 3.2(a to h), 3.3, and to Sections 4, 5,6,7,8,9,10.
3.2(j). Plan Ahead

We shall provide continuity benefit for listed Waiting Periods
served by the Policyholder (must be an Insured Person under
the Policy) to the newly married spouse or newborn child
added during the Policy Period.

i. First 30 days Waiting Period

ii. Specific Illness Waiting Period

iii. Pre-Existing Disease Waiting Period

Provided,

i. This benefit can be opted by the Insured at the time of new
Policy inception or at any renewal.
```

</details>

---

## `g-006` · lookup · starhealth starhealth__health__comprehensive.pdf · p.31

**Q:** How long do I have to wait before I can get coverage for something I was already sick with when I bought the policy?

**A:** The waiting period for any pre-existing disease would be reduced to the extent of prior coverage, and coverage is subject to the condition being declared and accepted at the time of application, after 36 months

*verified: False · vocab overlap: 0.2*

<details><summary>source chunk</summary>

```
on portability stipulated by IRDAI,
then waiting period for the same
would be reduced to the extent of
prior coverage
D.
Coverage under the policy after the
expiry of 36 months for any preexisting disease is subject to the
same being declared at the time of
application and accepted by Insurer.
2.
Specified disease/procedure waiting
period – Code Excl 02
A.
Expenses related to the treatment
of the listed Conditions, surgeries/
treatments shall be excluded until
the expiry of 24 months of continuous
coverage after the date of inception
of the first policy with us. This
exclusion shall not be applicable for
claims arising due to an accident.
B.
In case of enhancement of Sum Insured
the exclusion shall apply afresh to the
extent of Sum Insured increase
C. If any of the specified disease/
procedure falls under the waiting
period
specified
for
pre-Existing
diseases, then the longer of the two
waiting periods shall apply.
```

</details>

---

## `g-007` · lookup · starhealth starhealth__health__comprehensive.pdf · p.30

**Q:** Is my cataract surgery covered right away?

**A:** No, there is a waiting period for Treatment of Cataract

*verified: False · vocab overlap: 0.2*

<details><summary>source chunk</summary>

```
arising due to an accident
B.
In case of enhancement of Sum
Insured the exclusion shall apply
afresh to the extent of Sum Insured
increase
C. If any of the specified disease/
procedure falls under the waiting
period
specified
for
pre-Existing
diseases, then the longer of the two
waiting periods shall apply
D.
The waiting period for listed conditions
shall apply even if contracted after
the policy or declared and accepted
without a specific exclusion
E.
List of specific diseases/procedures;
i.
Treatment
of
Cataract
and
diseases of the anterior and
posterior chamber of the Eye,
Diseases of ENT, Diseases related
to Thyroid, Benign diseases of
the breast
ii.
Subcutaneous Benign Lumps,
Sebaceous cyst, Dermoid cyst,
Mucous cyst lip / cheek, Carpal
Tunnel
Syndrome,
Trigger
Finger, Lipoma, Neurofibroma,
Fibroadenoma, Ganglion and
similar pathology
iii.
All treatments (Conservative,
Operative treatment) and all types
of intervention for Diseases related
to Tendon, Ligament, Fascia, Bones
```

</details>

---

## `g-008` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.26

**Q:** What documents do I need to provide for a claim related to my bank account?

**A:** Cancelled cheque or copy of first page of bank passbook showing account holder’s name, Account number, IFSC code, Branch name etc.

*verified: False · vocab overlap: 0.571*

<details><summary>source chunk</summary>

```
Passport / Election Card, etc)
for address mentioned in Claim
form with KYC Form
11. Beneficiary bank account /
NEFT details: Cancelled cheque
or copy of first page of bank
passbook showing account
holder’s name, Account
number, IFSC code, Branch
name etc.
12. Certified copy of Death
certificate issued by municipal
authority (in case of death of
insured)
13. KYC details and Documents
Essential
Covers
Road Ambulance
Air Ambulance
Radio Cab
Organ Donor
Modern Treatments
Home Health Care
Consumables Cover
Restore Benefit
1. Same Documents as
mentioned in Section-
Hospitalization Cover
Special Covers
Convalescence
Companion Cover
Adventure Sports
Gym and Sports
Injury
Reconstructive
Surgery
Prosthetics
Gender
Reassignment
Surgery
1. Same Documents as
mentioned in
Section-Hospitalization Cover
2. All consultation bills and
prescriptions of Sports
Specialist/Medical Practitioner
3. Diagnostic test bills along with
copy of reports
4. Physiotherapy bills
5. Travel, food and
```

</details>

---

## `g-009` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.24

**Q:** What needs to happen before you will pay out under my policy?

**A:** The due payment of premium and realization thereof by the insurance company and the observance and fulfilment of the terms, provisions, conditions and endorsements of the policy

*verified: False · vocab overlap: 0.167*

<details><summary>source chunk</summary>

```
ii. Upon exhaustion of Sum Insured and Cumulative Bonus,
for the Policy year. However, the Policy is subject to

Policy shall be paid in accordance with the schedule of
payments in the Policy Schedule agreed between the
Policyholder and Us in writing. No receipt for premium shall be
valid except on Our official

form signed by Our duly authorized official. The due payment
of premium and realization thereof by Us and the observance
and fulfilment of the terms, provisions, conditions and
endorsements of this Policy by the Insured Person in so far as
they relate to anything to be done or complied with by the
Insured Person shall be a Condition Precedent to Our liability
to make any payment under this Policy.
14.1(i). Territorial Jurisdiction
```

</details>

---

## `g-010` · lookup · starhealth starhealth__health__comprehensive.pdf · p.35

**Q:** What happens if I get hurt while flying, but I'm not a paying passenger?

**A:** The claim would be excluded, specifically Code Sec10 Excl 03 if it's due to an accident I caused, but the exact exclusion code isn't specified in this case

*verified: False · vocab overlap: 0.125*

<details><summary>source chunk</summary>

```
such other similar aids - Code Excl 35
33. Any
hospitalizations
which
are
not
Medically Necessary / does not warrant
Hospitalization - Code Excl 36
34. Other Excluded Expenses as detailed
in
the
website
www.starhealth.in
- Code Excl 37
35. Existing
disease/s,
disclosed
by
the insured and mentioned in the
policy
schedule
under
Permanent
Exclusion (based on insured’s consent)
- Code Excl 38
B. Applicable for Section II.23
1.
Any claim relating to events occurring
before
the
commencement
of
the
cover or otherwise outside the Period of
Insurance - Code Sec10 Excl 01
2.
Any injuries/conditions which are Preexisting conditions - Code Sec10 Excl 02
3.
Any claim arising out of Accidents that
the Insured Person has caused - Code
Sec10 Excl 03
i.
intentionally or
ii.
by committing a crime / involved in it or
iii.
as a result of / in a state of drunkenness
or addiction (drugs, alcohol).
4.
Insured Person engaging in Air Travel
unless he/she flies as a fare-paying
```

</details>

---

## `g-011` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.18

**Q:** Is angioplasty considered heart surgery under my policy?

**A:** No, angioplasty is excluded.

*verified: False · vocab overlap: 0.6*

<details><summary>source chunk</summary>

```
3.
Open Chest CABG

The actual undergoing of heart Surgery to correct blockage or
narrowing in one or more coronary artery(s), by coronary
artery bypass grafting done via a sternotomy (cutting through
the breastbone) or minimally invasive keyhole coronary artery
bypass procedures. The diagnosis must be supported by a
coronary angiography and the realization of Surgery has to be
confirmed by a cardiologist.

The following are excluded:

Angioplasty and/or any other intra-arterial procedures.
4.
Open Heart Replacement or Repair of Heart Valves
```

</details>

---

## `g-012` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.23

**Q:** How much notice will I get if my health insurance product is being discontinued?

**A:** 90 days prior to expiry of the Policy

*verified: False · vocab overlap: 0.429*

<details><summary>source chunk</summary>

```
SBI General Insurance Company Limited
SBI General Insurance Company Limited, Corporate & Registered Office: Fulcrum Building, 9th Floor, A & B Wing, Sahar Road, Andheri (East), Mumbai - 400099. |
India and used by SBI General Insurance Company Limited under license | IRDAI Reg No: 144 | SBI General Health Alpha, UIN: SBIHLIP26038V012526 | SBI General
Insurance and SBI are separate legal entities and SBI is working as Corporate Agent of the company for sourcing of insurance products.
14.1(c). Withdrawal of the Product

i. ln the likelihood of this product being withdrawn in future,

the Company will intimate the Insured Person about the
same 90 days prior to expiry of the Policy.

ii. lnsured Person will have the option to migrate to similar
health insurance product available with the Company at
the time of renewal with all the accrued continuity benefits
such as Cumulative Bonus, waiver of Waiting Period as per
```

</details>

---

## `g-013` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.10

**Q:** What safety gear do I need to wear when riding an all-terrain vehicle?

**A:** helmets, face shields, goggles, protective gloves and footwear and clothing

*verified: False · vocab overlap: 0.5*

<details><summary>source chunk</summary>

```
2. The hot air Balloon used for the
expedition should have certified as
"Airworthy" by respective Civil Aviation
Authority.
3. Only tethered hot air ballooning is
covered under the Policy.
12
All Terrain
Vehicle tours
1. The guide overseeing the operations
should have been certified on driving
training
course
either
from
the
European ATV safety institute or the All
Terrain Safety Institute.
2. The participants must be wearing
prescribed protective equipment’s of
recommended quality such as (not
limited to) helmets, face shields,
goggles,
protective
gloves
and
footwear
and
clothing
as
recommended for the operation of the
ATV or quad bikes.
13
3.2(d). Gym and Sports Injury Cover
```

</details>

---

## `g-014` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.14

**Q:** What happens to my personal accident coverage if I make a claim for permanent total disability?

**A:** My personal accident coverage will terminate, but if I made a previous claim for permanent partial disability, the amount payable for the subsequent claim will be reduced by the amount already paid.

*verified: False · vocab overlap: 0.556*

<details><summary>source chunk</summary>

```
iv. Personal Accident, if opted, shall terminate in the event of
a Claim in respect of that Insured Person, becomes
admissible and accepted by Us under benefit 4.1.
Accidental Death (AD) and/or 4.2 Permanent Total
Disablement (PTD). Except if Claim is paid under benefit
4.3. Permanent Partial Disablement, the amount payable
for the subsequent Claim/s under any benefit of Personal
Accident shall be reduced by the amount/s already paid.

v. In the event of Permanent Total Disablement, the Insured
will be under obligation to:

a. Have himself/herself examined by the Panel Doctors
appointed (at the sole discretion of Company) and We
will
pay
the
costs
involved
thereof;
Any
non-compliance to the same may result in rejection of
the Claims.

b. Registered and Qualified Medical Practitioner providing
treatment or giving expert opinion and any other
authority to supply Us any information that may be
required on the condition of the Insured.
```

</details>

---

## `g-015` · lookup · starhealth starhealth__health__comprehensive.pdf · p.17

**Q:** How often can my family get a health check-up under this policy if we have a long-term plan?

**A:** once every Policy Year

*verified: False · vocab overlap: 0.556*

<details><summary>source chunk</summary>

```
iii.
For the updated and applicable list of
tests available under such package,
Insured Persons are required to check
our website www.starhealth.in.
iv. The pre-defined health check-up
packages may be modified from
time to time without prior notice.
v.
This cover can be availed through
Star health mobile application, other
digital platforms, or by calling at our
Toll free number: 1800 425 2255.
vi. The Network Provider/Health Service
Provider shall be assigned by Us
upon receiving the Insured Person’s
request to avail a Health Check-up
under this cover.
vii. Utilization of this Health Check-up
shall not impact the Sum Insured.
viii. In case of long term policies, Insured
Person(s) are eligible for a Health
Check-up once every Policy Year.
19. E-Domestic Second Medical Opinion:
The
Insured
Person
can
obtain
a
Second Medical Opinion from a Doctor
in the Company’s network of Medical
Practitioners practicing in India. All the
medical records provided by the Insured
```

</details>

---

## `g-016` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.11

**Q:** How long do I have to wait before I can get coverage for laser eye surgery?

**A:** 12/24 months from the date of inception of my first policy

*verified: False · vocab overlap: 0.25*

<details><summary>source chunk</summary>

```
We shall indemnify the Insured Person up to an amount
specified in the Policy Schedule, for the Medical Expenses
incurred during the Policy Year, for undergoing Medically
Necessary Treatment Laser-Assisted In Situ Keratomileusis
(LASIK) Surgery, including refractive keratotomy (RK) and
photorefractive keratectomy (PRK) or any other advanced
Surgical Procedures conducted to correct the refractive
errors beyond +/- 4.5 dioptre to rectify the refraction of one
or both eyes, on the written advice of the Medical
Practitioner, provided:

i. This benefit shall become available only after the expiry of
12/24 months (as opted) from the date of inception of the
Insured Person’s first Policy with Us.

ii. We have accepted a Claim under any one of the following
benefits, 3(a). In-Patient Treatment or 3(b). Day Care
Treatment.

iii. The treatment carried out for the cosmetic reasons is
excluded.

iv. Pre-Hospitalization and Post-Hospitalization expenses
shall not be covered under this benefit.
```

</details>

---

## `g-017` · lookup · starhealth starhealth__health__comprehensive.pdf · p.45

**Q:** How can I get help if I have questions about my insurance?

**A:** You can contact Star Health and Allied Insurance Company Limited at their office during normal business hours.

*verified: False · vocab overlap: 0.2*

<details><summary>source chunk</summary>

```
of
the
company
for
necessary
compliance by all stake holders.
29. Customer Service: If at any time the
Insured Person requires any clarification
or assistance, the insured may contact
Star Health and Allied Insurance Company
Limited, “Balaji Complex, No.15, Whites
Lane, Whites Road, Royapettah, Chennai-
600014”, during normal business hours.
31. Third-Party Claims: Only the authorized
Third-Party Administrator (TPA), legal
heirs, Proposer, or approved Insurance
Intermediaries have the right to make
or follow up on reimbursement claims
under this policy. The claims will be settled
directly to the Proposer’s account. In case
of the Proposer’s death, the settlement
will be made to the nominee’s account
as the case may be.
Claims by unauthorized third party will
not be entertained, and the Company
reserves the right to reject such claims.
This rejection does not breach the terms
of the policy.
32. Professional Conduct: The Company shall
ensure that its staff and representatives
```

</details>

---

## `g-018` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.13

**Q:** How long can I get weekly benefits if I'm completely unable to work due to an accident?

**A:** for more than 104 weeks

*verified: False · vocab overlap: 0.444*

<details><summary>source chunk</summary>

```
If the Insured Person sustains an Injury in an Accident during
the Policy Period and which completely incapacitates the
Insured Person from engaging in any employment or
occupation of any description whatsoever which the Insured
Person was capable of performing at the time of the Accident
(Temporary Total Disablement), We shall pay weekly benefit
as specified in the Policy Schedule, till the time the Insured
Person is able to return to work, provided that:

i. We shall be liable to make payment under this benefit in
respect of the Insured Person, if the Temporary Total
Disablement shall exceed the minimum number of 30 days
as specified in the Policy Schedule, during the Policy
Period.

ii. The compensation under this benefit shall not be payable
for more than 104 weeks in respect of any one Injury
calculated
from
the
date
of
commencement
of
disablement and in no case shall exceed the Personal
Accident Sum Insured.
```

</details>

---

## `g-019` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.16

**Q:** Is surgery to fix a problem with the main artery that carries blood from my heart covered?

**A:** Yes, it is covered under Aorta Graft Surgery, point 5

*verified: False · vocab overlap: 0.222*

<details><summary>source chunk</summary>

```
(allogeneic bone marrow transplant).
Pulmonary
Artery Graft
Surgery
4
We will be covering the undergoing of
Surgery requiring median sternotomy on
the advice of a Cardiologist for disease of
the pulmonary artery to excise and replace
the diseased pulmonary artery with a
graft.
Aorta Graft
Surgery
5
I. We
will
be
covering
the
actual
undergoing of major Surgery to repair
or
correct
aneurysm,
narrowing,
obstruction or dissection of the Aorta
through surgical opening of the chest
or abdomen. For the purpose of this
cover the definition of “Aorta” shall
mean the thoracic and abdominal aorta
but not its branches.
II. The following are excluded:
```

</details>

---

## `g-020` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.22

**Q:** Is my surgery to change my gender covered by the insurance?

**A:** Expenses related to any treatment, including surgical management, to change characteristics of the body to those of the opposite sex are excluded, but this exclusion does not apply if it is to comply with the Transgender Persons (Protection of Rights) Act, 2019.

*verified: False · vocab overlap: 0.6*

<details><summary>source chunk</summary>

```
b. greater than or equal to 35 in conjunction with any of
the following severe co-morbidities following failure of
less invasive methods of weight loss:

•
Obesity-related cardiomyopathy

•
Coronary heart disease

•
Severe Sleep Apnea

•
Uncontrolled Type2 Diabetes.
13.1(d). Change-of-Gender Treatments (Code: Excl07)

Expenses related to any treatment, including surgical
management, to change characteristics of the body to those
of the opposite sex. However, such exclusion shall not be
applicable to respective Insured Person to comply with
Transgender Persons (Protection of Rights) Act, 2019.
13.1(e).
Cosmetic or Plastic Surgery (Code: Excl08):
```

</details>

---

## `g-021` · lookup · starhealth starhealth__health__comprehensive.pdf · p.41

**Q:** How much time do I have to switch my policy to a different insurance company before my renewal date?

**A:** at least 30 days before, but not earlier than 60 days from the policy renewal date

*verified: False · vocab overlap: 0.4*

<details><summary>source chunk</summary>

```
Claim Bonus, Specific Waiting Periods,
Waiting period for Pre-Existing Diseases,
Moratorium period etc. in the previous
policy to the migrated policy.
8.
Portability:
i.
The Policyholder has the choice to
port his / her policy from one Insurer
to another by applying to such
Insurer to port the entire policy along
with all the members of the family, if
any, at least 30 days before, but not
earlier than 60 days from the policy
renewal date as per IRDAI guidelines
related to portability.
ii.
The Policyholder is entitled to transfer
the credits gained to the extent of
the Sum Insured, No Claim Bonus,
Specific Waiting Periods, Waiting
period for Pre-Existing Diseases,
Moratorium period etc. from the
existing Insurer to the Acquiring
Insurer in the previous policy.
9.
Renewal of policy: The policy shall
be renewable provided the product
is not withdrawn, except in case of
established fraud or non-disclosure or
misrepresentation by the Policyholder. If
```

</details>

---

## `g-022` · lookup · starhealth starhealth__health__comprehensive.pdf · p.14

**Q:** Is my weight loss surgery covered and how much will the insurance pay for it?

**A:** Expenses for weight loss surgery are payable, with a maximum limit of Rs.2,50,000/- and Rs.5,00,000/-, which includes pre-hospitalization and post-hospitalization expenses.

*verified: False · vocab overlap: 0.429*

<details><summary>source chunk</summary>

```
iii. This cover is available only when;
a.
both Self and Spouse are covered
under this policy either on floater
basis or on individual basis and both
Self and Spouse should have been
covered for a continuous period of 24
months under Star Comprehensive
Insurance Policy,
b.
the policy covering the self and
spouse are in force when the benefit
under this Section becomes payable.
iv. Claims under this Section
a.
will not reduce the Sum Insured;
b.
will affect Cumulative Bonus.
15. Bariatric Surgery: Expenses incurred
on hospitalization for bariatric surgical
procedure
and
its
complications
thereof are payable subject to limits
mentioned in the table given below,
during the Policy Period. This maximum
limit of Rs.2,50,000/- and Rs.5,00,000/-
are inclusive of pre-hospitalization and
post-hospitalization expenses.
```

</details>

---

## `g-023` · lookup · starhealth starhealth__health__comprehensive.pdf · p.40

**Q:** How much notice do I need to give to cancel my policy?

**A:** 7

*verified: False · vocab overlap: 0.333*

<details><summary>source chunk</summary>

```
Person or by his agent or the hospital/
doctor/any other party acting on behalf
of the Insured Person, with intent to
deceive the insurer or to induce the
insurer to issue an insurance policy:
i.
the suggestion, as a fact of that which
is not true and which the Insured Person
does not believe to be true;
ii.
the active concealment of a fact by the
Insured Person having knowledge or
belief of the fact;
iii.
any other act fitted to deceive; and
iv.
any such act or omission as the law
specially declares to be fraudulent
The Company shall not repudiate the claim
and / or forfeit the policy benefits on the
ground of Fraud, if the Insured Person /
beneficiary can prove that the misstatement
was true to the best of his knowledge and
there was no deliberate intention to suppress
the fact or that such misstatement of or
suppression of material fact are within the
knowledge of the insurer.
6.
Cancellation
i.
The Policy Holder may cancel his policy
any time during the term by giving 7
```

</details>

---

## `g-024` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.19

**Q:** If I don't use this benefit this year, can I use it next year?

**A:** No, it shall not be carried forward to any subsequent Policy Year

*verified: False · vocab overlap: 0.4*

<details><summary>source chunk</summary>

```
iv. If this benefit is not utilized in a Policy Year, it shall not be
carried forward to any subsequent Policy Year and it will be
the Insured Person’s choice and responsibility to utilize the
same with in the designated Policy Year. We shall not be
liable to provide any reminders or notifications for the
same.

v. In case of an Individual Policy, this benefit shall apply on
individual basis and in case of a Floater Policy, this benefit
shall apply on floater basis.
```

</details>

---

## `g-025` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.4

**Q:** What kind of doctor can treat my mental health issues and be recognized by my insurance?

**A:** A medical practitioner possessing a post-graduate degree or diploma in psychiatry awarded by an university recognized by the University Grants Commission established under the University Grants Commission Act, 1956, or awarded or recognized by the National Board of Examinations and included in the First Schedule to the Indian Medical Council

*verified: False · vocab overlap: 0.25*

<details><summary>source chunk</summary>

```
v. Is certified by the attending Medical Practitioner as a Life
Threatening Medical Condition.
2.2.(n). Mental Illness means a substantial disorder of thinking,
mood, perception, orientation or memory that grossly
impairs judgment, behavior, capacity to recognize reality or
ability to meet the ordinary demands of life, mental
conditions associated with the abuse of alcohol and drugs,
but does not include mental retardation which is a condition
of arrested or incomplete development of mind of a person,
specially characterized by sub normality of intelligence.
2.2.(o). Medical Practitioner for Mental Illnesses means a medical
practitioner possessing a post-graduate degree or diploma in
psychiatry awarded by an university recognized by the
University Grants Commission established under the
University Grants Commission Act, 1956, or awarded or
recognized by the National Board of Examinations and
included in the First Schedule to the Indian Medical Council
```

</details>

---

## `g-026` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.25

**Q:** What information do I need to provide to get pre-approval for my family's hospital treatment?

**A:** You need to provide 11 pieces of information, including policy number, name of the insured, nature of disease, and treatment details.

*verified: False · vocab overlap: 0.444*

<details><summary>source chunk</summary>

```
from
the
Hospital,
whichever
is
earlier
1. Policy Number
2. Name of the Insured
person(s) named in the
Policy schedule availing
treatment
3. Nature of
disease/Illness/Injury
4. Name and address of the
attending
5. Medical Practitioner/
Hospital
6. Date of admission &
probable date of discharge
7. Approximate Claim
Expenses
8. Treatment Details
9. Claim Form /
Pre-Authorization
Request form
10. KYC Form and KYC
Documents
11. Any other relevant
information as required
Not Applicable
Particulars
to be
provided
for
Pre-Autho
rization
1. If the particulars are not
provided in full or are
insufficient for us to
consider the request in
Pre-defined Claim Form,
We will request additional
information or
documentation
2. On receipt of duly filled
preauthorization form
from the Network Provider
along with other sufficient
Not Applicable
Process
for
obtaining
Pre-Autho
rization
```

</details>

---

## `g-027` · lookup · starhealth starhealth__health__comprehensive.pdf · p.5

**Q:** What kind of doctor can I see to get medical expenses covered for my family?

**A:** A person who holds a valid registration from the Medical Council of any State or Medical Council of India or Council for Indian Medicine or for Homeopathy set up by the Government of India or a State Government and is acting within the jurisdiction.

*verified: False · vocab overlap: 0.25*

<details><summary>source chunk</summary>

```
termination of pregnancy during the
Policy Period.
Medical Advice: Medical Advice means
any consultation or advice from a Medical
Practitioner including the issue of any
prescription or follow-up prescription.
Medical Expenses: Medical expenses means
those expenses that an Insured Person has
necessarily and actually incurred for medical
treatment on account of Illness or Accident on
the advice of a Medical Practitioner, as long
as these are no more than would have been
payable if the Insured Person had not been
insured and no more than other hospitals
or doctors in the same locality would have
charged for the same medical treatment.
Medical Practitioner: Medical Practitioner
is a person who holds a valid registration
from the Medical Council of any State or
Medical Council of India or Council for Indian
Medicine or for Homeopathy set up by the
Government of India or a State Government
and is thereby entitled to practice medicine
within its jurisdiction; and is acting within the
```

</details>

---

## `g-028` · lookup · starhealth starhealth__health__comprehensive.pdf · p.7

**Q:** What is considered when calculating associated medical expenses for my hospital treatment?

**A:** Associated medical expenses include nursing charges, operation theatre charges, and professional fees of medical practitioners, but do not include cost of pharmacy and consumables, cost of implants and medical devices, and cost of diagnostics, ICU charges.

*verified: False · vocab overlap: 0.571*

<details><summary>source chunk</summary>

```
Associated medical expenses: Associated
Medical Expenses means expenses that
shall include the applicable nursing charges,
Operation
theatre
charges,
Professional
fees
of
Medical
Practitioner
including
Surgeon/ anesthetist / Physician/Specialist
of the Hospital where the Insured Person
has been admitted and treated and hence
Proportionate deduction will be applicable
on these items.
“Associated Medical Expenses” does not
include cost of pharmacy and consumables,
cost of implants and medical devices
and cost of diagnostics, ICU charges and
hence proportionate deduction will not be
applicable on these items.
Basic Sum Insured: Basic Sum Insured means
the Sum Insured opted for and for which the
premium is paid.
Company / Insurer / We / Us: Company /
Insurer / We / Us means Star Health and
Allied Insurance Company Limited.
Dependent Child: Dependent Child means
a child (natural or legally adopted) who is
financially dependent and does not have his
or her independent source of income and
```

</details>

---

## `g-029` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.20

**Q:** What happens to my bonus if I switch from individual to family coverage when I renew my policy?

**A:** The bonus to be carried forward will be the lowest one that is applicable among all the family members.

*verified: False · vocab overlap: 0.5*

<details><summary>source chunk</summary>

```
Specific condition applicable to benefit 11.1. Cumulative Bonus

i. In case where the Policy is on individual basis as specified in
the Policy Schedule, the Cumulative Bonus shall be added
and available individually to the Insured Person and in case
where the Policy is on floater basis, the Cumulative Bonus
shall be added and available to the Family on floater basis.

ii. Cumulative Bonus shall be available only if the Policy is
renewed/ premium paid within the Grace Period.

iii. If the Insured Persons in the expiring Policy are covered on
an individual basis as specified in the Policy Schedule and
there is an accumulated Cumulative Bonus for such
Insured Persons under the expiring Policy, and such
expiring Policy has been renewed on a floater Policy basis
as specified in the Policy Schedule then the Cumulative
Bonus to be carried forward for credit in such renewed
Policy shall be the lowest one that is applicable among all
the Insured Persons.
```

</details>

---

## `g-030` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.27

**Q:** What is the phone number I can call for customer care?

**A:** 18001021111

*verified: False · vocab overlap: 0.4*

<details><summary>source chunk</summary>

```
All Claims under the Policy shall be payable in Indian
currency only.
14.3(i). Sequence of Sum Insured Applicability under Section-3

In case of an admissible claim, the sequence of Sum Insured
applicability shall be:

i. Base Sum Insured

ii. Cumulative Bonus (if applicable)

iii. Endless Sum Insured ( if opted)/Restore Benefit(if opted)
14.3(j). Overriding effect of the Policy Schedule

In case of any inconsistency in the terms and conditions in
this Policy vis-a-vis the information contained in the Policy
Schedule, the information contained in the Policy Schedule
shall prevail.
14.4.
Conditions For Renewal of The Contract
14.4(a).
Migration
SBI General Insurance Company Limited
SBI General Insurance Company Limited, Corporate & Registered Office: Fulcrum Building, 9th Floor, A & B Wing, Sahar Road, Andheri (East), Mumbai - 400099. |
CIN: U66000MH2009PLC190546 | Tollfree: 18001021111 | customer.care@sbigeneral.in | www.sbigeneral.in | SBI Logo displayed belongs to State Bank of
```

</details>

---

## `g-031` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.6

**Q:** Will my emergency transportation to the hospital be covered if I need to go to a different hospital for treatment?

**A:** Yes, if prescribed by a Medical Practitioner and is medically necessary, and the transportation is from one hospital to another that can provide the necessary medical services.

*verified: False · vocab overlap: 0.429*

<details><summary>source chunk</summary>

```
i. We have accepted a Claim under any one of the benefits,
Section 3 Hospitalization Cover, 3.1(d). Organ Donor or
3.1(e). Modern Treatments.

ii. The coverage includes the cost of the transportation of
the Insured Person to the nearest Hospital in case of an
emergency Life Threatening Medical condition, or from
one Hospital to another Hospital which is prepared to
admit the Insured Person and provide the necessary
medical services.

iii. Such Life-Threatening Medical Condition is certified by
the Medical Practitioner.

iv. The transportation from one Hospital to another Hospital
has been prescribed by a Medical Practitioner and is
medically necessary.

v. The Ambulance service is offered by a healthcare or
registered ambulance service provider.
3.1(b). Air Ambulance

We will indemnify the Insured Person up to the limit specified
in the Policy Schedule, for the expenses incurred on availing
air ambulance services during the Policy Year, provided:
```

</details>

---

## `g-032` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.8

**Q:** Will my insurance cover me if I get hurt while doing adventure sports like biking or rafting after drinking alcohol?

**A:** No, the insurance will not cover expenses incurred while participating in adventure sports under the influence of alcohol.

*verified: False · vocab overlap: 0.357*

<details><summary>source chunk</summary>

```
i. Zip Lining

ii. Bungee Jumping

iii. Parasailing

iv. Water Scooter rides

v. Speed Boat rides (not as an operator)

vi. Rafting

vii. Scuba Diving

viii. Snorkelling

ix. Trekking

x. Biking including Cycling and Motor Biking

xi. Hot Air Ballooning (Tethered)

xii. All-Terrain Vehicle tours

xiii. Personal Light Electric Vehicle (Segway/PLEV) tours

xiv. River Canoeing/Kayaking
Specific Exclusions applicable to benefit 3.2(c). Adventure Sports
We shall not be liable to make any payment under this benefit in
connection with or in respect of any expenses whatsoever incurred
by the Insured Person for:

i. Participation in any Adventure Sports whilst being under
influence of Alcohol or any other narcotic drugs or abuse
of prescription drugs or any hallucinates.

ii. Whilst being under any medication or treatment which
slows down response and alertness or makes the Insured
Person unfit for participating in such sports
```

</details>

---

## `g-033` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.28

**Q:** What phone number can I call for help?

**A:** 022-45138021

*verified: False · vocab overlap: 0.25*

<details><summary>source chunk</summary>

```
Phone: 022-45138021

Stage 4: Escalation to Insurance Ombudsman

If you feel that the response to your Grievance was
unsatisfactory, or if you believe your concerns have not
been adequately addressed by the company, you may
escalate the matter to the Insurance Ombudsman.

Submit your Grievance online:

https://www.cioins.co.in/Ombudsman
SBI General Insurance Company Limited
SBI General Insurance Company Limited, Corporate & Registered Office: Fulcrum Building, 9th Floor, A & B Wing, Sahar Road, Andheri (East), Mumbai - 400099. |
CIN: U66000MH2009PLC190546 | Tollfree: 18001021111 | customer.care@sbigeneral.in | www.sbigeneral.in | SBI Logo displayed belongs to State Bank of
```

</details>

---

## `g-034` · lookup · starhealth starhealth__health__comprehensive.pdf · p.3

**Q:** What is a congenital anomaly that I was born with and can be seen on my body?

**A:** External Congenital Anomaly

*verified: False · vocab overlap: 0.6*

<details><summary>source chunk</summary>

```
Cashless Facility: Cashless Facility means a
facility extended by the insurer to the insured
where the payments, of the cost of treatment
undergone by the insured in accordance with
the Policy Terms and conditions, are directly
made to the network provider by the insurer
to the extent pre-authorization approved.
Condition Precedent: Condition Precedent
means a Policy Term or condition upon
which the insurer’s liability under the policy is
conditional upon.
Congenital Anomaly: Congenital Anomaly
means a condition which is present since
birth, and which is abnormal with reference
to form, structure or position.
i.
Internal Congenital Anomaly: Congenital
anomaly which is not in the visible and
accessible parts of the body
ii.
External Congenital Anomaly: Congenital
anomaly which is in the visible and
accessible parts of the body
Co-payment: Co-payment means a costsharing requirement under a health insurance
policy that provides that the policyholder/
```

</details>

---

## `g-035` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.3

**Q:** How old do I have to be to be considered a senior citizen for my insurance?

**A:** sixty years or above

*verified: False · vocab overlap: 0.6*

<details><summary>source chunk</summary>

```
specific provider and consistent with the prevailing charges in
the geographical area for identical or similar services, taking
into account the nature of the Illness / Injury involved.
2.1.(ax). Renewal means the terms on which the contract of
insurance can be renewed on mutual consent with a provision
of Grace Period for treating the Renewal continuous for the
purpose of gaining credit for Pre-Existing Diseases,
time-bound exclusions and for all Waiting Periods.
2.1.(ay). Room Rent means the amount charged by a Hospital
towards Room and Boarding expenses and shall include the
associated Medical Expenses.
2.1.(az). Senior Citizen means any person, who has attained the
Age of sixty years or above.
2.1.(aaa). Solicitation means the act of approaching a prospect or a
Policyholder by an Insurer or by a distribution channel with a
view to persuading the prospect or a Policyholder to purchase
or to renew an insurance Policy.
2.1.(aab). Specific Waiting Period means a period up to 24 months
```

</details>

---

## `g-036` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.21

**Q:** How long do I have to wait before the policy covers me for diseases I already have?

**A:** 24 months of continuous coverage after the date of inception of the first Policy

*verified: False · vocab overlap: 0.286*

<details><summary>source chunk</summary>

```
i. Surgery to be conducted is upon the advice of the Doctor.

ii. The surgery/Procedure conducted should be supported by
clinical protocols

iii. The member has to be 18 years of age or older and

iv. Body Mass Index (BMI);

a. greater than or equal to 40 or

for the same would be reduced to the extent of prior
coverage.

iv. Coverage under the Policy after expiry of 24 months for
any Pre-Existing Disease (PED) is subject to the same
being declared at the time of application and accepted by
Us.
12.2. Specified
Diseases
and
Procedures
Waiting
Period
(Code-Excl 02):

i. Expenses related to the treatment of listed Conditions;
Surgeries/treatments shall be excluded until the expiry of
24 months of continuous coverage after the date of
inception of the first Policy with Us. This exclusion shall not
be applicable for Claims arising due to an accident.

ii. In case of enhancement of Sum Insured the exclusion shall
apply afresh to the extent of Sum Insured increase.
```

</details>

---

## `g-037` · lookup · starhealth starhealth__health__comprehensive.pdf · p.13

**Q:** What happens to my bonus if I make a claim and only use part of my insurance coverage?

**A:** The cumulative bonus will be reduced at the same rate at which it has accrued

*verified: False · vocab overlap: 0.222*

<details><summary>source chunk</summary>

```
Where the Sum Insured under the policy
is Rs.7,50,000/-or above, the Insured
Person would be entitled to the benefit
of Cumulative Bonus calculated at 100%
of the Basic Sum Insured under this
policy following a claim free year. The
maximum benefit of cumulative bonus
is 100% of the Basic Sum Insured.
Claims under Sections II.1, II.2, II.3, II.4, II.5,
II.6, II.7, II.8, II.9, II.10, II.11, II.14, II.15 and II.25 will
impact the eligibility and accumulation
of Cumulative Bonus.
Special Conditions for Cumulative Bonus
i.
The Cumulative Bonus will be calculated
on the expiring Basic Sum Insured.
ii.
If the insured opts to reduce the Basic
Sum Insured at the subsequent renewal,
the limit of indemnity by way of such
Cumulative Bonus shall not exceed such
reduced Basic Sum Insured.
iii. In the event of a claim resulting in
a.
Partial
utilization
of
Basic
Sum
Insured, such cumulative bonus so
granted will be reduced at the same
rate at which it has accrued
b.
```

</details>

---

## `g-038` · lookup · starhealth starhealth__health__comprehensive.pdf · p.28

**Q:** How many points do I get for doing a health check-up?

**A:** 200

*verified: False · vocab overlap: 0.5*

<details><summary>source chunk</summary>

```
Lets look how the Insured can avail discount on premium through the “Star Wellness Program”
Scenario – 2
A 42 year old Individual Suresh and his wife Lakshmi along with their two dependent children (aged
below 18 yrs) buy a Star Comprehensive Insurance Policy (Floater Sum Insured) with Sum Insured 25
Lacs, let’s understand how they can earn Wellness Points under the Floater Policy. Suresh has de­
clared that he is suffering from Diabetes & Hypertension. Suresh has declared his Body Mass Index
(BMI) as 30 & Lakshmi has declared her BMI as 25
Suresh and Lakshmi enrolled under the Star wellness program and completed the following wellness
activities.
Sl. No
Name of the wellness activity taken up during the
Policy Year
Wellness
Points Earned
by Suresh
Wellness
Points Earned
by Lakshmi
1.
Completed Online Health Risk Assessment (HRA)
50
50
2.
Submitted Health Check-Up Report
200
200
3.
Participation in Marathon
100
0
4.
Attended to Gym
100
100
5.
```

</details>

---

## `g-039` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.17

**Q:** What conditions will my insurance cover for kidney transplant surgery?

**A:** End stage renal disease presenting as chronic irreversible failure of both kidneys to function

*verified: False · vocab overlap: 0.5*

<details><summary>source chunk</summary>

```
iii. Arterial blood gas analysis with
partial
oxygen
pressure
of
55mmHg
or
less
(PaO2
<
55mmHg); and

iv. Dyspnea at rest.
Kidney
Transplant
Surgery in case
of End Stage
Renal Failure
9
We will be covering Kidney Transplant
Surgery due to following cases:
I. End stage renal disease presenting as
chronic irreversible failure of both
kidneys to function, as a result of which
either
regular
renal
dialysis
(haemodialysis or peritoneal dialysis) is
instituted or renal transplantation is
carried out. Diagnosis has to be
confirmed by a specialist medical
practitioner.
Surgery for
Pheochromocy
-toma
11
I. We
will
be
covering
the
actual
undergoing of Surgery to remove the
tumour.
II. Presence of a neuroendocrine tumour of
the adrenal or extra-chromaffin tissue
that secretes excess catecholamines and
the Diagnosis of Pheochromocytoma
must be confirmed by a Registered
Doctor who is an endocrinologist.
Surgical
Treatment of
Coma
10
I. We will be covering surgical treatment
of Coma limited to:
```

</details>

---

## `g-040` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.15

**Q:** How old do I have to be to get this benefit?

**A:** up to the age of 65 years

*verified: False · vocab overlap: 0.333*

<details><summary>source chunk</summary>

```
x. If Voluntary Co-Payment is opted, then it shall be
applicable on each and every Claim made under this
benefit.
We will indemnify the Insured Person, up to the limit specified in the
Policy Schedule, towards Medically Necessary Expenses incurred for
In-Patient Treatment or Day Care Treatment or OPD Treatment
including Planned Hospitalization incurred outside India and
anywhere across the world, during the Policy Year, provided:

i. This benefit can only be availed by Insured Person(s) up to
the age of 65 years.

ii. This benefit shall become available only after the expiry of
24 months from the date of inception of the first Policy
with Us except for Emergency Care.

iii. In-Patient Hospitalization, Day Care Procedure or
Out-Patient treatment, taken as Emergency Care shall be
covered up to the Sum Insured specified in the Policy
Schedule, provided the same is critical and cannot be
deferred till the Insured Person's return to the Republic of
India.
```

</details>

---

## `g-041` · lookup · sbigeneral sbigeneral__health__alpha.pdf · p.2

**Q:** What happens if I don't tell the truth or leave out important information when applying for insurance?

**A:** The policy can be affected in the event of misrepresentation, mis-description or non-disclosure of any material fact.

*verified: False · vocab overlap: 0.0*

<details><summary>source chunk</summary>

```
event
of
misrepresentation,
mis-description
or
non-disclosure of any material fact.
2.1(u). Domiciliary Hospitalization means medical treatment for an
Illness/disease/Injury which in the normal course would require
care and treatment at a Hospital but is actually taken while
confined at home under any of the following circumstances:
```

</details>

---

## `g-042` · lookup · starhealth starhealth__health__comprehensive.pdf · p.22

**Q:** How much do I get for completing a stress relief program if I don't have any chronic health issues?

**A:** 125

*verified: False · vocab overlap: 0.364*

<details><summary>source chunk</summary>

```
Insured who is not Overweight / Obese)
50
5.
a. Chronic Condition Management Program
(for the Insured who is suffering from Chronic
Condition/s - Diabetes, Hypertension,
Cardiovascular Disease or Asthma)
250
b. On Completion of De-Stress & Mind Body Healing
Program (for the Insured who is not suffering from
Chronic Condition/s - Diabetes, Hypertension,
Cardiovascular Disease or Asthma)
125
Additional Wellness Services
6.
Online Chat with Doctor
7.
Medical Concierge Services
8.
Period & Fertility Tracker
9.
Digital Health Vault
10.
Wellness Content
11.
Health Quiz & Gamification
12.
Post-Operative Care
13.
Discounts from Network Providers
```

</details>

---

## `g-043` · lookup · starhealth starhealth__health__comprehensive.pdf · p.36

**Q:** Will my insurance pay if I get hurt while doing something illegal?

**A:** No

*verified: False · vocab overlap: 0.25*

<details><summary>source chunk</summary>

```
actual or attempted commission of or
willful participation in an illegal act or
any violation or attempted violation of
the law - Code Sec10 Excl 11
12. Any payment in case of more than one
claim under the policy during the period
of insurance by which the maximum
liability of the Company in that period
would exceed the amount specified in
the Schedule - Code Sec10 Excl 12
13. Any other claim after a claim has been
admitted by the Company and becomes
payable for Death or Permanent Total
Disablement, as mentioned in Table -
Code Sec10 Excl 13
14. Any claim arising out of an accident
related
to
pregnancy
or
childbirth,
infirmity, whether directly or indirectly -
Code Sec10 Excl 14
15. Any claim for Death or Permanent Total
Disablement of the Insured Person
from self-endangerment unless in selfdefense or to save human life - Code
Sec10 Excl 15
IV - CONDITIONS
S T A N D A R D C O N D I T I O N S
1.
Disclosure of Information: The policy
shall be void and all premium paid
```

</details>

---

## `g-044` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.5

**Q:** How many of my family members can be covered under this policy?

**A:** up to 6 members

*verified: False · vocab overlap: 0.6*

<details><summary>source chunk</summary>

```
This cover will be applicable for maximum up to 6 members.

In the event of the unfortunate death of the insured, the
Personal Accident cover shall continue for the Family member(s)
until expiry of the policy.
7. Accidental Damage Cover – General Contents

What we will cover

Accidental damage external or internal to general contents
including DG Set, Pump set, Solar Panel or in-house lift which can
be defined as loss or damage caused by a sudden, unintended or
unexpected event that is not the result of a deliberate act.

What we will not cover

i.
An intentional or deliberate act or any consequential loss or
damage that results from intentional or deliberate act

ii.
the process of professional cleaning, repairing, restoring or
renovation

iii.
a computer virus or computer malfunction

iv.
loss or corruption of any electronic data or files

v.
the scorching or burning by a cigar, cigarette or pipe

vi.
any construction, renovation, alteration or extension work
```

</details>

---

## `g-045` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.16

**Q:** Is my business covered if it's interrupted because I need to make changes to my equipment?

**A:** Any loss or damage with respect to any additional time required for making change(s) to the buildings, structures, or equipment

*verified: False · vocab overlap: 0.25*

<details><summary>source chunk</summary>

```
subjects and/or a rising against a sovereign government or other
authority.
“Mutiny” shall mean a willful resistance by members of legally armed
or peace-keeping forces to a superior officer.
3. EXCLUSIONS
This cover DOES NOT INDEMNIFY AGAINST:
For Material Damage:
1. Any loss arising from War (whether before or after the outbreak
of hostilities) between any two or more countries.
2. Third party liability howsoever arising except as may be insured
specifically under any Third-Party Liability Extension to this
cover.
For Loss of Profit/Business Interruption:
1
Any loss or damage during any period in which goods would not
have been produced, or Operations or services would not have
been maintained, for any reason other than physical loss or
physical damage of the type insured against to which this
coverage applies.
2
Any loss or damage with respect to any additional time required
for making change(s) to the buildings, structures, or equipment
```

</details>

---

## `g-046` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.9

**Q:** How soon do I need to tell the police if something is stolen from me after something bad happens to my property?

**A:** within 7 (seven) days

*verified: False · vocab overlap: 0.3*

<details><summary>source chunk</summary>

```
iii. You must not carry out repairs, unless such repairs
are urgent and You cannot contact Us.

3. Immediate notice to Authorities

a. As soon as any loss or damage occurs to the Insured
Property, You must give immediate report to
appropriate legal authorities. For example, You must
report to the fire brigade of the local authority and
the police if there is damage by fire/ explosion /
implosion or lightning. In case of subsidence
/landslide/rockslide, You must inform the District
Administration. In the event of impact damage of any
kind or Riot Strikes, Malicious damages and acts of
terrorism, You must inform the police. If there is a
theft within 7 (seven) days following an Insured Event
You must inform the police.

b. We may, but not necessarily, waive this condition if
We are satisfied that by reason of extreme hardship
it was not possible for You or any other person on
Your behalf to give such report.

4. Submit claim:

a. Claim form:
```

</details>

---

## `g-047` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.8

**Q:** Can I cancel my insurance at any time and if so, what happens to the money I've paid?

**A:** You can cancel at any time and get a refund of the proportion premium for the unexpired policy period, if it's up to one year and you haven't made a claim.

*verified: False · vocab overlap: 0.429*

<details><summary>source chunk</summary>

```
II. Renewal of Policy

1. End of Policy: This Policy will expire at the end of the
Policy Period.

2. Renewal is not automatic: We may seek relevant
information from You for the purpose of renewal. We
can
reject
Your
renewal
only
on
grounds
of
misrepresentation, non-disclosure of material facts,
fraud or non-co-operation on Your part.

3. Application for renewal: If You wish to renew the Policy,
You must apply for renewal before the end of the Policy
Period and pay the required premium amount.

III. Cancellation and Termination of Policy

1. Cancellation by You

a. You can cancel this Policy at any time during the
policy period by giving Us notice in writing, in such
case, We shall

(i) refund the proportion premium for unexpired policy
period, if the period/term of the policy is up to one
year and there is no claim(s) made during the policy
period.
```

</details>

---

## `g-048` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.2

**Q:** What kind of things are considered valuable contents in my home insurance?

**A:** items such as jewellery, silverware, paintings, works of art, antique items, curios and items of similar nature

*verified: False · vocab overlap: 0.571*

<details><summary>source chunk</summary>

```
We give insurance cover for physical loss or damage, or destruction
caused to Insured Property by the following unforeseen events
occurring during the Policy Period.
The events covered are given in Column A and those not covered in
respect of these events are given in Column B.
Clause B. Insured Events
Valuable
Contents
Valuable Contents of Your Home consist of
items such as jewellery, silverware, paintings,
works of art, antique items, curios and items of
similar nature.
Insured
The
Person/s
who
has/have
purchased
Insurance Cover under this Policy.
Spouse
Your wife or husband.
Sum Insured
The amount shown as Sum Insured in the Policy
Schedule and as described in Clause C (4) and
Clause D (2) of this Policy. It represents Our
maximum liability for each cover or part of cover
and for each loss.
Solicitation
“Solicitation” means the act of approaching a
prospect or a policyholder by an insurer or by a
distribution channel with a view to persuade the
```

</details>

---

## `g-049` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.10

**Q:** What happens to the money you recover from someone who caused my loss?

**A:** Any amount recovered will be applied first to the costs of the legal proceedings and recovery, then to the claim amount paid or to be paid to me, and I will get any balance.

*verified: False · vocab overlap: 0.333*

<details><summary>source chunk</summary>

```
This Policy will be subject to the laws of India, and to the
jurisdiction of courts in India.
Clause K. Grievances
If You have a grievance about any matter relating to the Policy, or
Our decision on any matter, or the claim, You can address Your
grievance as follows:

this purpose. You must not do anything which will
prejudice Our right. We can do this

i.
without seeking Your consent,

ii.
in Your name, and

iii. whether
or
not
Your
loss
has
been
fully
compensated.

b. Any amount We recover from such person will be
applied first to the costs of the legal proceedings and
recovery, then to the claim amount We have paid or
must pay to You. We will pay You any balance.
```

</details>

---

## `g-050` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.4

**Q:** How long can I get help with lost rent if my home is damaged and I have to repair it?

**A:** The maximum period of this cover is three years from the date my home becomes unfit for living.

*verified: False · vocab overlap: 0.5*

<details><summary>source chunk</summary>

```
iii. The amount of lost rent shall be calculated as follows:

iv. Sum Insured for Cover for Loss of Rent (as declared by You in
the Proposal Form and specified by Us in the Policy
Schedule) X Period necessary for repairs ÷ Loss of Rent
Period opted for.

v.
This cover will be available for the reasonable time required
to repair Your Home Building to make it fit for living. The
maximum period of this cover is three years from the date
Your Home Building becomes unfit for living. You must
submit a certificate from an architect or the local authority
to show that Your Home Building is not fit for living.

vi. Claim for loss of rent will be accepted only if We have
accepted Your claim for loss for physical damage to Your
Home under the Home Building Cover.

as given in Clause E (5) of this Policy.
```

</details>

---

## `g-051` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.6

**Q:** If my air conditioner catches fire because of a short circuit, will the insurance cover the cost of the air conditioner?

**A:** No, the cost of the air conditioner will not be considered in the claim amount.

*verified: False · vocab overlap: 0.4*

<details><summary>source chunk</summary>

```
Subject otherwise to terms, conditions, limitations and
exceptions of the Policy.
11. Electrical Clause / Electrical Installation Clause

The policy covers loss or damage by fire to electrical appliance
and installation insured by this policy arising from or occasioned
by overrunning, excessive pressure, short circuit, arcing,

self-heating or leakage of electricity from whatever cause
(lightning included) subject to a maximum of Rs. 1lac

Provided that no liability exists under this Policy for loss or
damage to any electrical machine, apparatus, fixture or fittings
or to any portion of the electrical installation unless caused by
fire and allied perils as covered under the policy.

Subject otherwise to terms, conditions, limitations and
exceptions of the Policy.

Example:

If there is short circuit in AC resulting in spread of fire, cost of AC
will not be considered in the claim amount as per the exclusion
no 5 in the policy wording.
```

</details>

---

## `g-052` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.7

**Q:** How soon do I need to call the insurance company after discovering a loss?

**A:** within 24 hrs

*verified: False · vocab overlap: 0.5*

<details><summary>source chunk</summary>

```
e. You shall call Us at 1800 22 1111 or 1800 102 1111 or provide
written intimation within 24 hrs of discovering the loss to
make a claim and obtain the proper forms and instructions.

f.
You may file a police report within 24 hrs of discovering a
covered incident

g. You shall fit out and return any claims forms and
accompanying documents including police report (where
necessary), receipts for replacing locks and/or keys, and any
other documents We may ask You to provide.

h. The claim form and accompanying documents must be
returned to Us within 3 days of making the original claim

IV. Duties After an Accident or Loss In the event of a covered
loss:

a) You shall call Us at 1800-22-1111 or provide written
intimation within 24 hrs. of discovering the loss to make a
claim and obtain the proper forms and instructions;

b) You shall file a police report within 24 hours of discovering a
covered incident.
```

</details>

---

## `g-053` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.13

**Q:** What is considered a civil war under my policy?

**A:** an internecine war, or a war carried on between or among opposing citizens of the same country or nation

*verified: False · vocab overlap: 0.5*

<details><summary>source chunk</summary>

```
4. Civil War.

Such perils in respect of which cover has been purchased by the
Insured shall be the "Covered Causes of Loss".
2. DEFINITIONS

"Civil Commotion" shall mean any act committed in the course of
a disturbance of the public peace (where such disturbance is
motivated by political reasons) by any person taking part
together with others in such disturbance or any act of any
lawfully constituted authority for the purpose of suppressing or
minimising the consequence of such act.

"Civil War" shall mean an internecine war, or a war carried on
between or among opposing citizens of the same country or
nation.

"Coup d'Etat" shall mean the sudden, violent and illegal
overthrow of a sovereign government or any attempt at such
overthrow.
```

</details>

---

## `g-054` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.11

**Q:** Is mold damage covered under my policy?

**A:** No, loss or damage directly or indirectly caused by mould is not covered.

*verified: False · vocab overlap: 0.25*

<details><summary>source chunk</summary>

```
cessation, fluctuation or variation in, or insufficiency of, water,
gas or electricity supplies and telecommunications or any type
of service;
13. loss or increased cost as a result of threat or hoax;
14. loss or damage caused by or arising out of burglary, house -
breaking, looting, theft, larceny or any such attempt or any
omission of any kind of any person (whether or not such act is
committed in the course of a disturbance of public peace) in any
action taken in respect of an act of sabotage and/or terrorism;
15. loss or damage caused by mysterious disappearance or
unexplained loss;
16. loss or damage directly or indirectly caused by mould, mildew,
fungus, spores or other micro-organism of any type, nature or
description, including but not limited to any substance whose
presence poses an actual or potential threat to human health;
17. total or partial cessation of work or the retardation or
interruption or cessation of any process or operations or
omissions of any kind;
```

</details>

---

## `g-055` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.12

**Q:** What's the minimum amount I have to pay out of my own pocket if I make a claim for my shop?

**A:** INR 10,000

*verified: False · vocab overlap: 0.333*

<details><summary>source chunk</summary>

```
insurers, shall be INR 20,000,000,000. If the actual aggregate loss
suffered
at
one
compound/location
is
more
than
INR
20,000,000,000, the amounts payable towards individual policies
shall be reduced in proportion to the sum insured of the policies.
EXCESS*
Shops & Residential Risks: 1% of the claim amount for each and every
claim subject to Minimum of INR 10,000 and Maximum of INR
500,000
Non-Industrial Risks: 1% of the claim amount for each and every
claim subject to Minimum of INR 25,000 and Maximum of INR
1,000,00
Industrial Risks: 5% of the claim amount for each and every claim
subject to Minimum of INR 100,000 and Maximum of INR 25,00,000
*Whichever is applicable
ADD ON COVERS
It is further declared and agreed that the limit of indemnity including
the claim on add on cover(s) shall not exceed total sum insured plus
separate sublimit opted for add on cover(s) or INR 20,000,000,000
whichever is lower. In respect of several insurance policies within the
```

</details>

---

## `g-056` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.1

**Q:** When does my insurance coverage start?

**A:** It is the date and time shown in the Policy Schedule.

*verified: False · vocab overlap: 0.333*

<details><summary>source chunk</summary>

```
These words with special meaning are stated in the Policy with
the first letter in capitals.
SBI General Insurance Company Limited
Word/s
Bank
Specific meaning
A bank or any financial institution
Carpet Area
1. for the main building unit of Your Home, it is
the net usable floor area, excluding the area
covered by the external walls, areas under
services shafts, exclusive balcony or
It is the date and time from which the insurance
cover under this Policy begins. It is shown in the
Policy Schedule.
Commencem
ent Date
The amount required to construct Your Home
Building at the Commencement Date.
This amount is calculated as follows:
a. For residential structure of Your Home
including Fittings and Fixtures:
```

</details>

---

## `g-057` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.14

**Q:** What is considered an act of terrorism under my insurance policy?

**A:** An act or series of acts, including but not limited to the use of force or violence and/or the threat thereof, committed for political, religious, ideological or similar purposes.

*verified: False · vocab overlap: 0.4*

<details><summary>source chunk</summary>

```
For the purpose of this cover, an act of Terrorism means an act or
series of acts, including but not limited to the use of force or violence
and/or the threat thereof, of any person or group(s) of persons,
whether acting alone or on behalf of or in connection with any
organization(s) or government(s), or unlawful associations,
recognized under Unlawful Activities (Prevention) Act, 1967 (as
amended from time to time) or any other related and applicable
national or state legislation formulated to combat unlawful and
terrorist activities in the nation for the time being in force,
committed for political, religious, ideological or similar purposes
including the intention to influence any government and/or to put
the public or any section of the public in fear for such purposes.
For the purpose of this cover, an act of sabotage means a subversive
act or series of such acts committed for political, religious or
ideological purposes including an intention to influence any
```

</details>

---

## `g-058` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.15

**Q:** What is the toll-free number I can call for customer care?

**A:** 18001021111

*verified: False · vocab overlap: 0.333*

<details><summary>source chunk</summary>

```
SBI General Insurance Company Limited, Corporate & Registered Office: Fulcrum Building, 9th Floor, A & B Wing, Sahar Road, Andheri (East), Mumbai - 400099. |
CIN: U66000MH2009PLC190546 | Tollfree: 18001021111 | customer.care@sbigeneral.in | www.sbigeneral.in | SBI Logo displayed belongs to State Bank of
Add on UIN: IRDAN144RP0014V01202223/A0019V01202627 | SBI General Insurance and SBI are separate legal entities and SBI is working as Corporate Agent of
the company for sourcing of insurance products.
SBI General Insurance Company Limited
19. Any infidelity, fraudulent, dishonest or criminal act by any
director, officer or trustee of the Insured whether acting alone
or in collusion with others;
20. Any debt, insolvency or commercial failure, whether to provide
bond or security or otherwise, or any other financial cause of any
party or person whatsoever.
21. Loss or damage caused by Civil Commotion, Insurrection,
Revolution or Rebellion, Mutiny and/or Coup d’état and Civil War
```

</details>

---

## `g-059` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.3

**Q:** How long after something bad happens to my home can I still claim for something stolen?

**A:** within 7 (seven) days from the occurrence of and proximately caused by any of the above Insured Events

*verified: False · vocab overlap: 0.111*

<details><summary>source chunk</summary>

```
dispossession,
confiscation,
commandeering,
requisition or destruction
by order of the
government or any lawful
authority, or
b. temporary or permanent
dispossession of Your
Home by unlawful
occupation by any person.
Bursting or overflowing of
water tanks, apparatus and
pipes.
11.
-
Leakage from automatic
sprinkler installations.
12.
a. repairs or alterations in
Your Home or the building
in which Your home is
located,
b. repairs, removal or
extension of any sprinkler
installation, or defects in
the construction known to
You.
Theft within 7 (seven)days
from the occurrence of and
proximately caused by any of
the above Insured Events.
13.
if it is
a. of any article or thing
outside Your Home, or of
any article or thing
attached from the outside
of the outer walls or the
roof of Your Home, unless
securely mounted.
SBI General Insurance Company Limited, Corporate & Registered Office: Fulcrum Building, 9th Floor, A & B Wing, Sahar Road, Andheri (East), Mumbai - 400099. |
```

</details>

---

## `g-060` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.17

**Q:** What kinds of injuries are covered under my policy?

**A:** All physical injury to a third-party human being including death, sickness, disease or disability and all mental injury, anguish or shock to such human being resulting from such physical injury.

*verified: False · vocab overlap: 0.25*

<details><summary>source chunk</summary>

```
1.8. In the event the Insured elects not to appeal, a judgement which
may, in whole or in part, involve indemnity under this Policy,
Insurer may, following discussion with the Insured, elect to make
such appeal at their own cost and expense and shall be liable for
the taxable costs and disbursements and any additional interest
incidental to such appeal; but in no event shall the liability of
Insurer exceed the relevant limits of liability plus such cost,
expense, disbursements and interest.
2. Definition
The words “Bodily Injury”, wherever used in this policy, shall mean all
physical injury to a third-party human being including death,
sickness, disease or disability and all mental injury, anguish or shock
to such human being resulting from such physical injury.
3. Exclusions
1. Any loss arising from War (whether before or after the outbreak
of hostilities) between any two or more countries.
2. Loss, injury or damage arising out of discrimination or
humiliation.
```

</details>

---

## `g-061` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.5

**Q:** How many of my family members can be covered under this policy?

**A:** up to 6 members

*verified: False · vocab overlap: 0.6*

<details><summary>source chunk</summary>

```
This cover will be applicable for maximum up to 6 members.

In the event of the unfortunate death of the insured, the
Personal Accident cover shall continue for the Family member(s)
until expiry of the policy.
7. Accidental Damage Cover – General Contents

What we will cover

Accidental damage external or internal to general contents
including DG Set, Pump set, Solar Panel or in-house lift which can
be defined as loss or damage caused by a sudden, unintended or
unexpected event that is not the result of a deliberate act.

What we will not cover

i.
An intentional or deliberate act or any consequential loss or
damage that results from intentional or deliberate act

ii.
the process of professional cleaning, repairing, restoring or
renovation

iii.
a computer virus or computer malfunction

iv.
loss or corruption of any electronic data or files

v.
the scorching or burning by a cigar, cigarette or pipe

vi.
any construction, renovation, alteration or extension work
```

</details>

---

## `g-062` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.16

**Q:** Is my business covered if it's interrupted because of a war between countries?

**A:** No, any loss arising from War is not covered.

*verified: False · vocab overlap: 0.571*

<details><summary>source chunk</summary>

```
subjects and/or a rising against a sovereign government or other
authority.
“Mutiny” shall mean a willful resistance by members of legally armed
or peace-keeping forces to a superior officer.
3. EXCLUSIONS
This cover DOES NOT INDEMNIFY AGAINST:
For Material Damage:
1. Any loss arising from War (whether before or after the outbreak
of hostilities) between any two or more countries.
2. Third party liability howsoever arising except as may be insured
specifically under any Third-Party Liability Extension to this
cover.
For Loss of Profit/Business Interruption:
1
Any loss or damage during any period in which goods would not
have been produced, or Operations or services would not have
been maintained, for any reason other than physical loss or
physical damage of the type insured against to which this
coverage applies.
2
Any loss or damage with respect to any additional time required
for making change(s) to the buildings, structures, or equipment
```

</details>

---

## `g-063` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.9

**Q:** How soon do I need to tell the police if something is stolen from me after something bad happens to my property?

**A:** You must inform the police within 7 days following the event.

*verified: False · vocab overlap: 0.3*

<details><summary>source chunk</summary>

```
iii. You must not carry out repairs, unless such repairs
are urgent and You cannot contact Us.

3. Immediate notice to Authorities

a. As soon as any loss or damage occurs to the Insured
Property, You must give immediate report to
appropriate legal authorities. For example, You must
report to the fire brigade of the local authority and
the police if there is damage by fire/ explosion /
implosion or lightning. In case of subsidence
/landslide/rockslide, You must inform the District
Administration. In the event of impact damage of any
kind or Riot Strikes, Malicious damages and acts of
terrorism, You must inform the police. If there is a
theft within 7 (seven) days following an Insured Event
You must inform the police.

b. We may, but not necessarily, waive this condition if
We are satisfied that by reason of extreme hardship
it was not possible for You or any other person on
Your behalf to give such report.

4. Submit claim:

a. Claim form:
```

</details>

---

## `g-064` · lookup · sbigeneral sbigeneral__home__house-insurance.pdf · p.8

**Q:** Can I cancel my insurance at any time and if so, what happens to the money I've already paid?

**A:** You can cancel at any time and get a refund of the proportion premium for the unexpired policy period, if it's up to one year and you haven't made any claims.

*verified: False · vocab overlap: 0.375*

<details><summary>source chunk</summary>

```
II. Renewal of Policy

1. End of Policy: This Policy will expire at the end of the
Policy Period.

2. Renewal is not automatic: We may seek relevant
information from You for the purpose of renewal. We
can
reject
Your
renewal
only
on
grounds
of
misrepresentation, non-disclosure of material facts,
fraud or non-co-operation on Your part.

3. Application for renewal: If You wish to renew the Policy,
You must apply for renewal before the end of the Policy
Period and pay the required premium amount.

III. Cancellation and Termination of Policy

1. Cancellation by You

a. You can cancel this Policy at any time during the
policy period by giving Us notice in writing, in such
case, We shall

(i) refund the proportion premium for unexpired policy
period, if the period/term of the policy is up to one
year and there is no claim(s) made during the policy
period.
```

</details>

---

## `g-065` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.8

**Q:** How long can my life insurance be questioned if something important was left out or incorrectly stated when I applied?

**A:** within 3 years

*verified: False · vocab overlap: 0.273*

<details><summary>source chunk</summary>

```
keeping silence to speak or silence is in itself equivalent to speak. 5. No Insurer
shall repudiate a life insurance Policy on the ground of Fraud, if the Insured /
beneficiary can prove that the misstatement was true to the best of his knowledge
and there was no deliberate intention to suppress the fact or that such misstatement of or suppression of material fact are within the knowledge of the
insurer. Onus of disproving is upon the policyholder, if alive, or beneficiaries. 6. Life
insurance Policy can be called in question within 3 years on the ground that any
statement of or suppression of a fact material to expectancy of life of the insured
was incorrectly made in the proposal or other document basis which policy was
issued or revived or rider issued. For this, the insurer should communicate in
writing to the insured or legal representative or nominee or assignees of insured,
as applicable, mentioning the ground and materials on which decision to repudiate
```

</details>

---

## `g-066` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.2

**Q:** What happens to my policy if I don't pay my overdue premiums within two years?

**A:** I will have two options: convert the policy into a paid-up policy or surrender the policy and receive the Fund Value, after which the policy will terminate.

*verified: False · vocab overlap: 0.444*

<details><summary>source chunk</summary>

```
continue as per the policy terms and conditions.
= If the overdue premiums are not paid before
the end of the two year revival period, then you
will have the following two options: iv. a.
Convert the policy into a paid-up policy. The
treatment thereafter will be as described in
option (iii) above. iv. b. Surrender the policy and
receive the Fund Value, at the end of the revival
period. On payment of the Fund Value this
policy shall terminate and all rights, benefits and
interests under this policy shall be extinguished.
No option is selected before the
end of the notice period
Treatment will be as if option ii were selected.
```

</details>

---

## `g-067` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.1

**Q:** Can someone I've chosen to manage my policy make changes to it after I'm gone?

**A:** No, the Nominee cannot make any policy transactions

*verified: False · vocab overlap: 0.25*

<details><summary>source chunk</summary>

```
alterations will be allowed. The Nominee cannot make any policy transactions such
as making partial withdrawals, paying top up premiums, performing switches,
renewing Automatic Transfer Strategy (ATS), redirecting premium, effecting a
change in portfolio strategy, opting for settlement option, increasing or decreasing
premium payment term, increasing or decreasing Sum Assured, increasing or
decreasing policy term. = Loyalty Additions and Wealth Boosters, as described in
Section 1.3 and Section 1.4 respectively, will continue to be allocated to the Fund
Value. iv. Death Benefit may be taxable as per prevailing tax laws.
```

</details>

---

## `g-068` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.6

**Q:** What kind of documents do I need to give to the insurance company if my family member dies in an accident?

**A:** Copy of First Investigation Report (FIR), post mortem, panchnama, final police investigation report etc.

*verified: False · vocab overlap: 0.3*

<details><summary>source chunk</summary>

```
the following documents: = ClaimantEs Statement = Original policy document
= Death Certificate of the Life Assured issued by the local municipal authority
and medical authority = Copy of First Investigation Report (FIR), post mortem,
panchnama, final police nvestigation report etc. in case of death due to accident =
Copy of all medical tests/ records, admission records, discharge summary,
prescriptions etc where death is not due to accident = Any other documents or
information as may be required by the Company for processing of the claim
depending on the cause of the death. Claim payments are made only in Indian
currency in accordance with the prevailing Exchange control regulations and
other relevant laws and regulations in India.
```

</details>

---

## `g-069` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.5

**Q:** What law applies to how I name someone to get my policy money if something happens to me?

**A:** Section 39 of the Insurance Act, 1938

*verified: False · vocab overlap: 0.111*

<details><summary>source chunk</summary>

```
plan, We will cancel the Policy from inception and refund the Fund Value less
premium discontinuance charge and the policy will terminate thereafter b) If the
Correct Age of the Life Assured makes him eligible for this Policy, revised Mortality
Charges per Part E will be payable as per the Correct Age from the next Policy
anniversary. There could be a revision in the Sum Assured also depending on the
correct age of the Life Assured. This section will be as per the provisions of Section
45 of the Insurance Act, 1938, as amended from time to time.
2. Nomination Nomination will be as per Section 39 of the Insurance Act, 1938.
Please refer to Annexure III for details on this section.
3. Assignment Assignment will be as per Section 38 of the Insurance Act, 1938.
Please refer to Annexure IV for details on this section.
4. Incontestability Incontestability will be as per Section 45 of the Insurance Act,
1938. Please refer Annexure V for more details on this section.
```

</details>

---

## `g-070` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.7

**Q:** How long after my life insurance policy starts can it be questioned for any reason?

**A:** 3 yrs

*verified: False · vocab overlap: 0.556*

<details><summary>source chunk</summary>

```
Provisions regarding policy not being called into question in terms of Section 45 of
the Insurance Act, 1938, as amended by Insurance Laws (Amendment) Ordinance
dtd 26.12.2014 are as follows: 1. No Policy of Life Insurance shall be called in
question on any ground whatsoever after expiry of 3 yrs from a. the date of
issuance of policy or b. the date of commencement of risk or c. the date of revival of
policy or d. the date of rider to the policy whichever is later. 2. On the ground of
fraud, a policy of Life Insurance may be called in question within 3 years from a. the
date of issuance of policy or b. the date of commencement of risk or c. the date of
revival of policy or d. the date of rider to the policy whichever is later. For this, the
insurer should communicate in writing to the insured or legal representative or
nominee or assignees of insured, as applicable, mentioning the ground and
materials on which such decision is based. 3. Fraud means any of the following
```

</details>

---

## `g-071` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.4

**Q:** How will my investments be changed in the last few years before my policy ends?

**A:** The exposure in the Multi Cap Growth Fund will be systematically reduced during the last ten quarters of the Policy term by automatic switches to the Income Fund.

*verified: False · vocab overlap: 0.25*

<details><summary>source chunk</summary>

```
Under this strategy, you have the option to make Partial Withdrawals. Partial
Withdrawals and different growth rates of the Multi Cap Growth and Income Fund
may cause the actual fund weightings to differ from the above schedule. Since the
objective is to allocate assets based on risk appetite at the current age, the
Policyholder funds will be regularly rebalanced to achieve the above allocations.
This will be done by automatic switching of units between the two funds at every
policy quarter. During the last ten quarters of the Policy term, the exposure in the
Multi Cap Growth Fund will be systematically reduced as per the PolicyholderEs
age as described in the table below by automatic switches to the Income Fund.
This is done so that the Fund Value at the time of maturity is not adversely affected
by short term volatility in the equity market that Multi Cap Growth Fund invests in.
```

</details>

---

## `g-072` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.8

**Q:** How long can my life insurance be questioned if something was incorrectly stated when I applied?

**A:** within 3 years

*verified: False · vocab overlap: 0.375*

<details><summary>source chunk</summary>

```
keeping silence to speak or silence is in itself equivalent to speak. 5. No Insurer
shall repudiate a life insurance Policy on the ground of Fraud, if the Insured /
beneficiary can prove that the misstatement was true to the best of his knowledge
and there was no deliberate intention to suppress the fact or that such misstatement of or suppression of material fact are within the knowledge of the
insurer. Onus of disproving is upon the policyholder, if alive, or beneficiaries. 6. Life
insurance Policy can be called in question within 3 years on the ground that any
statement of or suppression of a fact material to expectancy of life of the insured
was incorrectly made in the proposal or other document basis which policy was
issued or revived or rider issued. For this, the insurer should communicate in
writing to the insured or legal representative or nominee or assignees of insured,
as applicable, mentioning the ground and materials on which decision to repudiate
```

</details>

---

## `g-073` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.2

**Q:** What happens to my policy if I don't pay my overdue premiums within two years?

**A:** I will have two options: convert the policy into a paid-up policy or surrender the policy and receive the Fund Value, after which the policy will terminate.

*verified: False · vocab overlap: 0.444*

<details><summary>source chunk</summary>

```
continue as per the policy terms and conditions.
= If the overdue premiums are not paid before
the end of the two year revival period, then you
will have the following two options: iv. a.
Convert the policy into a paid-up policy. The
treatment thereafter will be as described in
option (iii) above. iv. b. Surrender the policy and
receive the Fund Value, at the end of the revival
period. On payment of the Fund Value this
policy shall terminate and all rights, benefits and
interests under this policy shall be extinguished.
No option is selected before the
end of the notice period
Treatment will be as if option ii were selected.
```

</details>

---

## `g-074` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.1

**Q:** Can someone I've chosen to manage my policy make changes to it after I'm gone?

**A:** No, the Nominee cannot make any policy transactions

*verified: False · vocab overlap: 0.25*

<details><summary>source chunk</summary>

```
alterations will be allowed. The Nominee cannot make any policy transactions such
as making partial withdrawals, paying top up premiums, performing switches,
renewing Automatic Transfer Strategy (ATS), redirecting premium, effecting a
change in portfolio strategy, opting for settlement option, increasing or decreasing
premium payment term, increasing or decreasing Sum Assured, increasing or
decreasing policy term. = Loyalty Additions and Wealth Boosters, as described in
Section 1.3 and Section 1.4 respectively, will continue to be allocated to the Fund
Value. iv. Death Benefit may be taxable as per prevailing tax laws.
```

</details>

---

## `g-075` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.6

**Q:** What kind of documents do I need to give to the insurance company if my family member dies in an accident?

**A:** Copy of First Investigation Report (FIR), post mortem, panchnama, final police investigation report etc.

*verified: False · vocab overlap: 0.3*

<details><summary>source chunk</summary>

```
the following documents: = ClaimantEs Statement = Original policy document
= Death Certificate of the Life Assured issued by the local municipal authority
and medical authority = Copy of First Investigation Report (FIR), post mortem,
panchnama, final police nvestigation report etc. in case of death due to accident =
Copy of all medical tests/ records, admission records, discharge summary,
prescriptions etc where death is not due to accident = Any other documents or
information as may be required by the Company for processing of the claim
depending on the cause of the death. Claim payments are made only in Indian
currency in accordance with the prevailing Exchange control regulations and
other relevant laws and regulations in India.
```

</details>

---

## `g-076` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.1

**Q:** How long do I have to revive my policy after it's been discontinued?

**A:** two consecutive years from the date of discontinuance of the Policy

*verified: False · vocab overlap: 0.4*

<details><summary>source chunk</summary>

```
of Units at the prevailing NAV of the Funds offered in this policy, in case of partial
withdrawals, switches, surrender, maturity etc. 28. Regulator is the authority that has
regulatory jurisdiction and powers over the Company. Currently the Regulator is
Insurance Regulatory and Development Authority of India (IRDAI). 29. Regular Pay
means premiums need to be paid regularly throughout the Policy term. 30. Revival of
the Policy means restoration of Policy benefits. 31. Revival Period means the period of
two consecutive years from the date of discontinuance of the Policy, during which
period You are entitled to revive the Policy. 32. Risk Commencement Date means the
date as specified in the Policy Certificate, on which the insurance coverage under this
Policy commences. 33. Single Pay means premium needs to be paid once at the start
of the Policy. 34. Sum Assured means the amount specified in the Policy Certificate.
```

</details>

---

## `g-077` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.5

**Q:** What law applies to how I name someone to get my policy money if something happens to me?

**A:** Section 39 of the Insurance Act, 1938

*verified: False · vocab overlap: 0.111*

<details><summary>source chunk</summary>

```
plan, We will cancel the Policy from inception and refund the Fund Value less
premium discontinuance charge and the policy will terminate thereafter b) If the
Correct Age of the Life Assured makes him eligible for this Policy, revised Mortality
Charges per Part E will be payable as per the Correct Age from the next Policy
anniversary. There could be a revision in the Sum Assured also depending on the
correct age of the Life Assured. This section will be as per the provisions of Section
45 of the Insurance Act, 1938, as amended from time to time.
2. Nomination Nomination will be as per Section 39 of the Insurance Act, 1938.
Please refer to Annexure III for details on this section.
3. Assignment Assignment will be as per Section 38 of the Insurance Act, 1938.
Please refer to Annexure IV for details on this section.
4. Incontestability Incontestability will be as per Section 45 of the Insurance Act,
1938. Please refer Annexure V for more details on this section.
```

</details>

---

## `g-078` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.5

**Q:** How can I pay my first premium and what types of payments are accepted?

**A:** First premium deposit received by way of local cheque or pay order or demand drafts payable at par or outstation cheque or pay order or demand drafts

*verified: False · vocab overlap: 0.5*

<details><summary>source chunk</summary>

```
consultation with IRDAI. = The Company will make investments as per the fund
mandates given in section 8.1 however the company reserves the right to change
the exposure of all/any fund to money market to 100% in extreme situation
external to the Company keeping in view market conditions/political
situations/economic situations/war like situations/terror situations. The same will
be put back as per the base mandate once the situation has corrected. = Some
examples of such circumstance in above sections are:  When one or more stock
First premium deposit received by way
of local cheque or pay order or
demand drafts payable at par
First premium deposit received by way
of outstation cheque or pay order or
demand drafts
Renewal premiums received by way of
direct debit, Electronic Clearing
System (ECS), credit card, etc.
Renewal premiums received by way of
local Cheque or pay order or demand
draft payable at par
```

</details>

---

## `g-079` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.7

**Q:** How long after my life insurance policy starts can it be questioned for any reason?

**A:** 3 yrs

*verified: False · vocab overlap: 0.556*

<details><summary>source chunk</summary>

```
Provisions regarding policy not being called into question in terms of Section 45 of
the Insurance Act, 1938, as amended by Insurance Laws (Amendment) Ordinance
dtd 26.12.2014 are as follows: 1. No Policy of Life Insurance shall be called in
question on any ground whatsoever after expiry of 3 yrs from a. the date of
issuance of policy or b. the date of commencement of risk or c. the date of revival of
policy or d. the date of rider to the policy whichever is later. 2. On the ground of
fraud, a policy of Life Insurance may be called in question within 3 years from a. the
date of issuance of policy or b. the date of commencement of risk or c. the date of
revival of policy or d. the date of rider to the policy whichever is later. For this, the
insurer should communicate in writing to the insured or legal representative or
nominee or assignees of insured, as applicable, mentioning the ground and
materials on which such decision is based. 3. Fraud means any of the following
```

</details>

---

## `g-080` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.7

**Q:** What happens to the money from my life insurance if I die and the person I chose to get it dies before me?

**A:** The proceeds are payable to me or my heirs or legal representatives or holder of succession certificate.

*verified: False · vocab overlap: 0.6*

<details><summary>source chunk</summary>

```
to the extent of insurerEs or transfereeEs or assigneeEs interest in the policy. The
nomination will get revived on repayment of the loan. 10. The right of any creditor
to be paid out of the proceeds of any policy of life insurance shall not be affected by
the nomination. 11. In case of nomination by policyholder whose life is insured, if
the nominees die before the policyholder, the proceeds are payable to
policyholder or his heirs or legal representatives or holder of succession
certificate. 12. In case nominee(s) survive the person whose life is insured, the
amount secured by the policy shall be paid to such survivor(s). 13. Where the
policyholder whose life is insured nominates his a. parents or b. spouse or c.
children or d. spouse and children e. or any of them the nominees are beneficially
entitled to the amount payable by the insurer to the policyholder unless it is proved
that policyholder could not have conferred such beneficial title on the nominee
```

</details>

---

## `g-081` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.8

**Q:** How soon after an accident do I have to die for it to be considered an accidental death?

**A:** within 180 days of the occurrence of such Accident

*verified: False · vocab overlap: 0.429*

<details><summary>source chunk</summary>

```
1.1. Accident is a sudden, unforeseen and involuntary event caused by external
and visible means. 1.2. Accidental Death shall mean death: which is caused by
Bodily Injury resulting from an Accident and which occurs due to the said Bodily
Injury solely, directly and independently of any other causes and which occurs
within 180 days of the occurrence of such Accident 1.3. Bodily Injury means Injury
must be evidenced by external signs such as contusion, bruise and wound except
in cases of drowning and internal injury. 1.4. Policy means and includes the Policy
Document, the proposal form for insurance submitted by the policyholder, the
benefit illustration signed by the policyholder, the Policy Specifications, the first
premium receipt, any attached endorsements or supplements together with all the
addendums provided by the Company from time to time, the medical examinerEs
report and any other document/s called for by the Company and submitted by the
```

</details>

---

## `g-082` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.4

**Q:** How will my investments be changed in the last few years before my policy ends?

**A:** The exposure in the Multi Cap Growth Fund will be systematically reduced during the last ten quarters of the Policy term by automatic switches to the Income Fund.

*verified: False · vocab overlap: 0.25*

<details><summary>source chunk</summary>

```
Under this strategy, you have the option to make Partial Withdrawals. Partial
Withdrawals and different growth rates of the Multi Cap Growth and Income Fund
may cause the actual fund weightings to differ from the above schedule. Since the
objective is to allocate assets based on risk appetite at the current age, the
Policyholder funds will be regularly rebalanced to achieve the above allocations.
This will be done by automatic switching of units between the two funds at every
policy quarter. During the last ten quarters of the Policy term, the exposure in the
Multi Cap Growth Fund will be systematically reduced as per the PolicyholderEs
age as described in the table below by automatic switches to the Income Fund.
This is done so that the Fund Value at the time of maturity is not adversely affected
by short term volatility in the equity market that Multi Cap Growth Fund invests in.
```

</details>

---

## `g-083` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.2

**Q:** What happens if I take my own life within a year of getting this policy?

**A:** The policy will be void and only the Fund Value including Top-up Fund Value, if any, as available on the date of death, will be payable. No charges will apply after the date of death.

*verified: False · vocab overlap: 0.5*

<details><summary>source chunk</summary>

```
If the Life Assured, whether sane or insane, commits suicide for any reason
whatsoever within one year of the date of issuance of the policy, the policy shall be
void and only the Fund Value including Top-up Fund Value, if any, as available on
the date of death of the Life Assured, will be payable. No charges will apply after the
date of death. The policy will terminate on the said payment and all rights, benefits
and interests will stand extinguished. If the Life Assured, whether sane or insane,
commits suicide within one year from the date of revival, the policy shall be void
and only the Fund Value including Top-up Fund Value, if any, as available on the
date of death of the Life Assured will be payable. As such, in effect, no charges will
apply after the date of death. The policy will terminate on the said payment and all
rights, benefits and interests will stand extinguished.

PART - D
1. Freelook Period
```

</details>

---

## `g-084` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.5

**Q:** How do you decide what investments to make with my money?

**A:** You will select the investments at your sole discretion subject to the investment objectives of the Fund and the applicable regulations in this regard.

*verified: False · vocab overlap: 0.25*

<details><summary>source chunk</summary>

```
2.7 Valuation date Valuation date is any date on which the NAV is declared by us.

2.8 Valuation of the Funds Valuation of Funds is the determination of the value of
the underlying assets of the Funds. The valuation of the assets will be made as per
the valuation norms prescribed by the Regulator and implemented by us.

2.9 Investment of the Funds We will select the investments, in accordance with
board approved investment policy, including derivatives and units of mutual
Funds, of the Fund at our sole discretion subject to the investment objectives of the
Fund and the applicable regulations in this regard.
```

</details>

---

## `g-085` · lookup · iciciprulife iciciprulife__life__prusmart.pdf · p.4

**Q:** What percentage of my investment will go into equity and related securities if I choose the high risk option?

**A:** 100%

*verified: False · vocab overlap: 0.556*

<details><summary>source chunk</summary>

```
Low
50%
100%
Debt Instruments
Money Market & Cash
viii) Maximise India Fund : To
offer long term wealth
maximization by managing a
diversified equity portfolio,
predominantly comprising of
companies in NIFTY 50 &
NIFTY Junior indices.
SFIN : ULIF 136 11/20/14 MIF
105.
80%
0%
0%

High
100%
20%
20%
Equity & equity
related securities
Debt Instruments
Money Market & Cash
80%
0%
0%

High
100%
20%
20%
E q u i t y & e q u i t y
related securities
Debt Instruments
Money Market &
Cash
i) Multi Cap Growth Fund: To
generate superior long-term
returns from a diversified
portfolio of equity and equity
related instruments of large,
mid and small cap companies.
SFIN : ULIF 085 24/11/09
LMCapGro 105.

If you have chosen Regular Pay premium payment option:

AP: Annualised Premium; SP: Single Premium

FV: Fund Value excluding Top-up Fund Value, if any, on the Date of Discontinuance

No Discontinuance Charge is applicable for Top-up premiums.
```

</details>

---

## `g-086` · negative · no source document · — (negative)

**Q:** Does this policy pay out if my business is sued for professional negligence?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---

## `g-087` · negative · no source document · — (negative)

**Q:** Is my rental car covered when I travel overseas?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---

## `g-088` · negative · no source document · — (negative)

**Q:** Does this cover trip cancellation if I miss my connecting train?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---

## `g-089` · negative · no source document · — (negative)

**Q:** Does this cover my flight being cancelled while I'm abroad?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---

## `g-090` · negative · no source document · — (negative)

**Q:** What's the excess on my two-wheeler insurance claim?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---

## `g-091` · negative · no source document · — (negative)

**Q:** Is satellite launch failure covered under this policy?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---

## `g-092` · negative · no source document · — (negative)

**Q:** My car was damaged in a collision last week. How much will I get back?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---

## `g-093` · negative · no source document · — (negative)

**Q:** Am I insured for cancelled concert tickets?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---

## `g-094` · negative · no source document · — (negative)

**Q:** What's the payout if my fishing boat sinks?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---

## `g-095` · negative · no source document · — (negative)

**Q:** Is my shipping container covered if the cargo is lost at sea?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---

## `g-096` · negative · no source document · — (negative)

**Q:** What happens if my freight is delayed at customs?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---

## `g-097` · negative · no source document · — (negative)

**Q:** Does this policy cover legal costs if I'm taken to court over a car accident?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---

## `g-098` · negative · no source document · — (negative)

**Q:** Can I claim for crop failure after the monsoon flooded my fields?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---

## `g-099` · negative · no source document · — (negative)

**Q:** How much do I get if my drone crashes into someone's property?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---

## `g-100` · negative · no source document · — (negative)

**Q:** Does this include cover for my employees' workplace injuries?

**A:** That isn't covered in the policy documents you've uploaded.

*verified: True · vocab overlap: 0.0*

---
