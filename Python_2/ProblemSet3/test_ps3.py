#from ps3b import Patient, SimpleVirus, NoChildException
from ps3b import *
## test simulation
#from ps3b_precompiled_36 import *

import math
import random
import numpy as np

import pylab

random.seed(0)

# : SimpleVirus 1
testcase = 3
print("SimpleVirus 1: ")
virus1 = SimpleVirus(1.0, 0.0)
virus2 = SimpleVirus(0.0, 1.0)
myVirus = [virus1, virus2]
for i in range(10):
    print("v1.doesClear(): " + str(virus1.doesClear()))

# SimpleVirus 2
print("\nSimpleVirus 2: ")
virus1 = SimpleVirus(0.0, 0.0)
for i in range(10):
    print("v1.doesClear(): " + str(virus1.doesClear()))

# SimpleVirus 3
print("\nSimpleVirus 3: ")
virus1 = SimpleVirus(1.0, 1.0)
for i in range(10):
    print("v1.doesClear(): " + str(virus1.doesClear()))

# SimpleVirus 4
print("\nSimpleVirus 4: ")
virus1 = SimpleVirus(0.0, 1.0)
for i in range(10):
    print("v1.doesClear(): " + str(virus1.doesClear()))

# SimpleVirus 5
print("\nSimpleVirus 5: ")
virus1 = SimpleVirus(0.98, 0.54)
popDensity = 0.04
for i in range(10):
    try:
        virus1.reproduce(popDensity)
        print("Reproduced successfully")
    except NoChildException:
        print("NoChildException")

# Patient 1
print("\nPatient  1: ")
viruses = [
    SimpleVirus(0.77, 0.54),
    SimpleVirus(0.16, 0.99),
    SimpleVirus(0.93, 0.73),
    SimpleVirus(0.85, 0.53),
    SimpleVirus(0.32, 0.93)
]
P1 = Patient(viruses, 7)
print("P1.getTotalPop() = " + str(P1.getTotalPop()))

# Patient 2
print("\nPatient 2: ")
virus = SimpleVirus(1.0, 0.0)
patient = Patient([virus], 100)
for i in range(100):
    patient.update()
print("patient.getTotalPop() expected to be " + str(patient.getTotalPop()))

# Patient 3
print("\nPatient 3: ")
virus = SimpleVirus(1.0, 1.0)
patient = Patient([virus], 100)
for i in range(100):
    patient.update()
print("patient.getTotalPop() expected to be " + str(patient.getTotalPop()))

# Patient 4
print("\nPatient 4: ")
virus = SimpleVirus(0.7, 0.96)
patient = Patient([virus], 8)
for i in range(10):
    patient.update()
    print("len(P1.viruses) < maxPop?" + str(patient.getTotalPop() < patient.getMaxPop()))

# Problem 3: simulationWithoutDrug
numTrials = 10
maxViruses = 100
maxPop = 1000
maxBirthProb = 0.1
clearProb = 0.05
#simulationWithoutDrug(maxViruses, maxPop, maxBirthProb, clearProb, numTrials)
#simulationWithoutDrug(1, 10, 1.0, 0.0, 1)

# Test Resistant Virus
maxBirthProb = 0.1
clearProb = 0.05
mutProb = 0.1
resistanceList = {'DrugA':True, 'DrugB': False, 'DrugC': False, 'DrugD': True}
virus = ResistantVirus(maxBirthProb, clearProb, resistanceList, mutProb)
drug = 'DrugC'
print('Resistance to ' +drug + " :" + str(virus.isResistantTo(drug)))

## Test TreatedPatient
print("\nTreatPatient 1")
virus = ResistantVirus(1.0, 0.0, {}, 0.0)
patient = TreatedPatient([virus], 100)
for i in range(100):
    patient.update()
print("Treated Patient.getTotalPop() expected to be " + str(patient.getTotalPop()))

## Test TreatedPatient 2
print("\nTreatPatient 2")
virus = ResistantVirus(1.0, 1.0, {}, 0.0)
patient = TreatedPatient([virus], 100)
for i in range(100):
    patient.update()
print("Treated Patient.getTotalPop() expected to be " + str(patient.getTotalPop()))

# Test TreatedPatient 3
print("\nTreatPatient 3")
virus = ResistantVirus(1.0, 0.0, {}, 0.0)
patient = TreatedPatient([virus], 100)
print("adding Drug_A")
patient.addPrescription("Drug_A")
patient.getPrescriptions()
print("Total Prescription: " + str(len(patient.getPrescriptions())))
print("adding Drug_A again")
patient.addPrescription("Drug_A")
patient.getPrescriptions()
print("Total Prescription: " + str(len(patient.getPrescriptions())))

# Test TreatedPatient 4
print("\nTreat Patient 4")
patient = TreatedPatient([], 100)
drug_list = ["J", "M", "O", "C", "Q", "P", "N", "Y", "E", "A"]
for i in drug_list:
    print("Adding prescription Drug" + i)
    patient.addPrescription(i)
    print("Drug " +i + " in list " +str(i in patient.getPrescriptions()))
list2 = ["J", "M", "O", "C", "Q"]
for i in drug_list:
    print("Adding prescription Drug " + i)
    patient.addPrescription(i)
    print("Drug " +i + " in list " +str(i in patient.getPrescriptions()))
    print("len of prescription = " + str(len(patient.getPrescriptions())))

# Test TreatedPatient 5
virus1 = ResistantVirus(1.0, 0.0, {"drug1": True}, 0.0)
virus2 = ResistantVirus(1.0, 0.0, {"drug1": False, "drug2": True}, 0.0)
virus3 = ResistantVirus(1.0, 0.0, {"drug1": True, "drug2": True}, 0.0)
patient = TreatedPatient([virus1, virus2, virus3], 100)
listOfList = [["drug1"], ["drug2"], ["drug1", "drug2"], ["drug3"], ["drug1", "drug3"], ["drug1", "drug2", "drug3"]]
for drugList in listOfList:
    print("patient getResistPop of drugList : " + str(patient.getResistPop(drugList)))

# Test TreatedPatient 6
virus1 = ResistantVirus(1.0, 0.0, {"drug1": True}, 0.0)
virus2 = ResistantVirus(1.0, 0.0, {"drug1": False}, 0.0)
patient = TreatedPatient([virus1, virus2, virus3], 1000000)
patient.addPrescription("drug1")
for i in range(5):
    patient.update()
print("Expect resistant population to be :" + str(patient.getTotalPop()))

# Test SimulationWithDrug
numTrials = 10
maxViruses = 100
maxPop = 1000
maxBirthProb = 0.1
clearProb = 0.05
resistances = {'guttagonol': False}
mutProb = 0.005
simulationWithDrug(maxViruses, maxPop, maxBirthProb, clearProb, resistances, mutProb, numTrials)
#simulationWithDrug(1, 10, 1.0, 0.0, {}, 1.0, 5)


