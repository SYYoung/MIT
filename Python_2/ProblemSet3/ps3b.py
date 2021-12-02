# Problem Set 3: Simulating the Spread of Disease and Virus Population Dynamics 

import random
import pylab
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

''' 
Begin helper code
'''

class NoChildException(Exception):
    """
    NoChildException is raised by the reproduce() method in the SimpleVirus
    and ResistantVirus classes to indicate that a virus particle does not
    reproduce. You can use NoChildException as is, you do not need to
    modify/add any code.
    """

'''
End helper code
'''

#
# PROBLEM 1
#
class SimpleVirus(object):

    """
    Representation of a simple virus (does not model drug effects/resistance).
    """
    def __init__(self, maxBirthProb, clearProb):
        """
        Initialize a SimpleVirus instance, saves all parameters as attributes
        of the instance.        
        maxBirthProb: Maximum reproduction probability (a float between 0-1)        
        clearProb: Maximum clearance probability (a float between 0-1).
        """

        # TODO
        self.maxBirthProb = maxBirthProb
        self.clearProb = clearProb

    def getMaxBirthProb(self):
        """
        Returns the max birth probability.
        """
        # TODO
        return self.maxBirthProb

    def getClearProb(self):
        """
        Returns the clear probability.
        """
        # TODO
        return self.clearProb

    def doesClear(self):
        """ Stochastically determines whether this virus particle is cleared from the
        patient's body at a time step. 
        returns: True with probability self.getClearProb and otherwise returns
        False.
        """

        # TODO
        choice = [True, False]
        return np.random.choice(choice, p=[self.clearProb, 1-self.clearProb])

    
    def reproduce(self, popDensity):
        """
        Stochastically determines whether this virus particle reproduces at a
        time step. Called by the update() method in the Patient and
        TreatedPatient classes. The virus particle reproduces with probability
        self.maxBirthProb * (1 - popDensity).
        
        If this virus particle reproduces, then reproduce() creates and returns
        the instance of the offspring SimpleVirus (which has the same
        maxBirthProb and clearProb values as its parent).         

        popDensity: the population density (a float), defined as the current
        virus population divided by the maximum population.         
        
        returns: a new instance of the SimpleVirus class representing the
        offspring of this virus particle. The child should have the same
        maxBirthProb and clearProb values as this virus. Raises a
        NoChildException if this virus particle does not reproduce.               
        """

        # TODO
        if (popDensity >= 1.0):
            prob = 0.0
        else:
            prob = self.maxBirthProb * (1 - popDensity)
        choice = np.random.choice([True, False], p=[prob, 1 - prob])

        if (choice):
            return SimpleVirus(self.maxBirthProb, self.clearProb)
        else:
            raise NoChildException



class Patient(object):
    """
    Representation of a simplified patient. The patient does not take any drugs
    and his/her virus populations have no drug resistance.
    """    

    def __init__(self, viruses, maxPop):
        """
        Initialization function, saves the viruses and maxPop parameters as
        attributes.

        viruses: the list representing the virus population (a list of
        SimpleVirus instances)

        maxPop: the maximum virus population for this patient (an integer)
        """

        # TODO
        self.viruses = list(viruses)
        self.maxPop = maxPop

    def getViruses(self):
        """
        Returns the viruses in this Patient.
        """
        # TODO
        return self.viruses


    def getMaxPop(self):
        """
        Returns the max population.
        """
        # TODO
        return self.maxPop


    def getTotalPop(self):
        """
        Gets the size of the current total virus population. 
        returns: The total virus population (an integer)
        """

        # TODO
        return len(self.viruses)


    def update(self):
        """
        Update the state of the virus population in this patient for a single
        time step. update() should execute the following steps in this order:
        
        - Determine whether each virus particle survives and updates the list
        of virus particles accordingly.   
        
        - The current population density is calculated. This population density
          value is used until the next call to update() 
        
        - Based on this value of population density, determine whether each 
          virus particle should reproduce and add offspring virus particles to 
          the list of viruses in this patient.                    

        returns: The total virus population at the end of the update (an
        integer)
        """

        # TODO
        for i in range(self.getTotalPop()-1, -1, -1):
            if self.viruses[i].doesClear():
                self.viruses.pop(i)

        popDensity = self.getTotalPop()/float(self.maxPop)

        newList = []
        for i in (self.viruses):
            #newList.append(i)
            #newVirus = i.reproduce(popDensity)
            #if (newVirus != None):
            #    newList.append(newVirus)
            try:
                newVirus = i.reproduce(popDensity)
                newList.append(i)
                newList.append(newVirus)
            except NoChildException:
                newList.append(i)
        self.viruses = newList
        return self.getTotalPop()



#
# PROBLEM 2
#
def simulationWithoutDrug(numViruses, maxPop, maxBirthProb, clearProb,
                          numTrials):
    """
    Run the simulation and plot the graph for problem 3 (no drugs are used,
    viruses do not have any drug resistance).    
    For each of numTrials trial, instantiates a patient, runs a simulation
    for 300 timesteps, and plots the average virus population size as a
    function of time.

    numViruses: number of SimpleVirus to create for patient (an integer)
    maxPop: maximum virus population for patient (an integer)
    maxBirthProb: Maximum reproduction probability (a float between 0-1)        
    clearProb: Maximum clearance probability (a float between 0-1)
    numTrials: number of simulation runs to execute (an integer)
    """

    # TODO
    # 1. simulate the list of viruses
    viruses = []
    for i in range(numViruses):
        virus = SimpleVirus(maxBirthProb, clearProb)
        viruses.append(virus)

    # 2. instantiate the patient
    # patient = Patient(viruses, maxPop)

    # 3. simulate with defined time steps and defined trials
    numTimeStep = 300
    virusTimeStep = np.zeros(numTimeStep)
    for trial in range(numTrials):
        trialVirus = []
        patient = Patient(viruses, maxPop)
        for i in range(1, numTimeStep+1):
            patient.update()
            trialVirus.append(patient.getTotalPop())
        virusTimeStep = virusTimeStep + np.array(trialVirus)

    # 4. take the average
    virusTimeStep = virusTimeStep/numTrials
    answer = list(virusTimeStep)

    # 5. plot the number of virus vs time step
    pylab.plot(answer, label="SimpleVirus")
    pylab.title("SimpleVirus simulation")
    pylab.xlabel("Time Steps")
    pylab.ylabel("Average Virus Population")
    pylab.legend(loc="best")
    pylab.show()


#
# PROBLEM 3
#
class ResistantVirus(SimpleVirus):
    """
    Representation of a virus which can have drug resistance.
    """   

    def __init__(self, maxBirthProb, clearProb, resistances, mutProb):
        """
        Initialize a ResistantVirus instance, saves all parameters as attributes
        of the instance.

        maxBirthProb: Maximum reproduction probability (a float between 0-1)       

        clearProb: Maximum clearance probability (a float between 0-1).

        resistances: A dictionary of drug names (strings) mapping to the state
        of this virus particle's resistance (either True or False) to each drug.
        e.g. {'guttagonol':False, 'srinol':False}, means that this virus
        particle is resistant to neither guttagonol nor srinol.

        mutProb: Mutation probability for this virus particle (a float). This is
        the probability of the offspring acquiring or losing resistance to a drug.
        """

        # TODO
        SimpleVirus.__init__(self, maxBirthProb, clearProb)
        self.resistances = dict(resistances)
        self.mutProb = mutProb

    def getResistances(self):
        """
        Returns the resistances for this virus.
        """
        # TODO
        return dict(self.resistances)

    def getMutProb(self):
        """
        Returns the mutation probability for this virus.
        """
        # TODO
        return self.mutProb

    def isResistantTo(self, drug):
        """
        Get the state of this virus particle's resistance to a drug. This method
        is called by getResistPop() in TreatedPatient to determine how many virus
        particles have resistance to a drug.       

        drug: The drug (a string)

        returns: True if this virus instance is resistant to the drug, False
        otherwise.
        """
        
        # TODO
        isResistant = self.resistances.get(drug)
        return isResistant


    def reproduce(self, popDensity, activeDrugs):
        """
        Stochastically determines whether this virus particle reproduces at a
        time step. Called by the update() method in the TreatedPatient class.

        A virus particle will only reproduce if it is resistant to ALL the drugs
        in the activeDrugs list. For example, if there are 2 drugs in the
        activeDrugs list, and the virus particle is resistant to 1 or no drugs,
        then it will NOT reproduce.

        Hence, if the virus is resistant to all drugs
        in activeDrugs, then the virus reproduces with probability:      

        self.maxBirthProb * (1 - popDensity).                       

        If this virus particle reproduces, then reproduce() creates and returns
        the instance of the offspring ResistantVirus (which has the same
        maxBirthProb and clearProb values as its parent). The offspring virus
        will have the same maxBirthProb, clearProb, and mutProb as the parent.

        For each drug resistance trait of the virus (i.e. each key of
        self.resistances), the offspring has probability 1-mutProb of
        inheriting that resistance trait from the parent, and probability
        mutProb of switching that resistance trait in the offspring.       

        For example, if a virus particle is resistant to guttagonol but not
        srinol, and self.mutProb is 0.1, then there is a 10% chance that
        that the offspring will lose resistance to guttagonol and a 90%
        chance that the offspring will be resistant to guttagonol.
        There is also a 10% chance that the offspring will gain resistance to
        srinol and a 90% chance that the offspring will not be resistant to
        srinol.

        popDensity: the population density (a float), defined as the current
        virus population divided by the maximum population       

        activeDrugs: a list of the drug names acting on this virus particle
        (a list of strings).

        returns: a new instance of the ResistantVirus class representing the
        offspring of this virus particle. The child should have the same
        maxBirthProb and clearProb values as this virus. Raises a
        NoChildException if this virus particle does not reproduce.
        """

        # TODO
        # 1. check if the virus is resistant to all the drugs in the active list
        resist = True
        for drug in activeDrugs:
            resist = self.isResistantTo(drug)
            if not resist:
                break
        if not resist:
            raise NoChildException

        # 2. if it is resistant to all the drugs, proceed to reproduce with probability
        if (popDensity >= 1.0):
            prob = 0.0
        else:
            prob = self.maxBirthProb * (1 - popDensity)
        choice = np.random.choice([True, False], p=[prob, 1 - prob])

        if not choice:
            raise NoChildException

        # 3. if it reproduces, for each self.resistance drug, select if it will be the same as the parent
        newResistance = dict(self.getResistances())
        mutProb = self.mutProb
        for drug in newResistance.keys():
            choice = np.random.choice([newResistance[drug], not newResistance[drug]], p=[1-mutProb, mutProb])
            newResistance[drug] = choice
        # create a new instance
        return ResistantVirus(self.maxBirthProb, self.clearProb, newResistance, mutProb)
            

class TreatedPatient(Patient):
    """
    Representation of a patient. The patient is able to take drugs and his/her
    virus population can acquire resistance to the drugs he/she takes.
    """

    def __init__(self, viruses, maxPop):
        """
        Initialization function, saves the viruses and maxPop parameters as
        attributes. Also initializes the list of drugs being administered
        (which should initially include no drugs).              

        viruses: The list representing the virus population (a list of
        virus instances)

        maxPop: The  maximum virus population for this patient (an integer)
        """

        # TODO
        Patient.__init__(self, viruses, maxPop)
        self.prescription = []


    def addPrescription(self, newDrug):
        """
        Administer a drug to this patient. After a prescription is added, the
        drug acts on the virus population for all subsequent time steps. If the
        newDrug is already prescribed to this patient, the method has no effect.

        newDrug: The name of the drug to administer to the patient (a string).

        postcondition: The list of drugs being administered to a patient is updated
        """

        # TODO
        if (newDrug not in self.prescription):
            self.prescription.append(newDrug)


    def getPrescriptions(self):
        """
        Returns the drugs that are being administered to this patient.

        returns: The list of drug names (strings) being administered to this
        patient.
        """

        # TODO
        return list(self.prescription)


    def getResistPop(self, drugResist):
        """
        Get the population of virus particles resistant to the drugs listed in
        drugResist.       

        drugResist: Which drug resistances to include in the population (a list
        of strings - e.g. ['guttagonol'] or ['guttagonol', 'srinol'])

        returns: The population of viruses (an integer) with resistances to all
        drugs in the drugResist list.
        """

        # TODO
        viruses = self.getViruses()
        total = 0
        for virus in viruses:
            resist = True
            for drug in drugResist:
                if (not virus.isResistantTo(drug)):
                    resist = False
                    break
            if resist:
                total += 1
        return total


    def update(self):
        """
        Update the state of the virus population in this patient for a single
        time step. update() should execute these actions in order:

        - Determine whether each virus particle survives and update the list of
          virus particles accordingly

        - The current population density is calculated. This population density
          value is used until the next call to update().

        - Based on this value of population density, determine whether each 
          virus particle should reproduce and add offspring virus particles to 
          the list of viruses in this patient.
          The list of drugs being administered should be accounted for in the
          determination of whether each virus particle reproduces.

        returns: The total virus population at the end of the update (an
        integer)
        """

        # TODO
        for i in range(self.getTotalPop()-1, -1, -1):
            if self.viruses[i].doesClear():
                self.viruses.pop(i)

        popDensity = self.getTotalPop()/float(self.maxPop)

        newList = []
        for i in (self.viruses):
            try:
                newVirus = i.reproduce(popDensity, self.prescription)
                newList.append(i)
                newList.append(newVirus)
            except NoChildException:
                newList.append(i)
        self.viruses = newList
        return self.getTotalPop()



#
# PROBLEM 4
#
def simulationWithDrug(numViruses, maxPop, maxBirthProb, clearProb, resistances,
                       mutProb, numTrials):
    """
    Runs simulations and plots graphs for problem 5.

    For each of numTrials trials, instantiates a patient, runs a simulation for
    150 timesteps, adds guttagonol, and runs the simulation for an additional
    150 timesteps.  At the end plots the average virus population size
    (for both the total virus population and the guttagonol-resistant virus
    population) as a function of time.

    numViruses: number of ResistantVirus to create for patient (an integer)
    maxPop: maximum virus population for patient (an integer)
    maxBirthProb: Maximum reproduction probability (a float between 0-1)        
    clearProb: maximum clearance probability (a float between 0-1)
    resistances: a dictionary of drugs that each ResistantVirus is resistant to
                 (e.g., {'guttagonol': False})
    mutProb: mutation probability for each ResistantVirus particle
             (a float between 0-1). 
    numTrials: number of simulation runs to execute (an integer)
    
    """

    # TODO
    # 1. simulate the list of viruses
    viruses = []
    for i in range(numViruses):
        virus = ResistantVirus(maxBirthProb, clearProb, resistances, mutProb)
        viruses.append(virus)

    # 2. instantiate the patient
    drugToTestList = ['guttagonol']

    # 3. simulate with defined time steps and defined trials
    numTimeStep = 150
    virusTimeStep = np.zeros(numTimeStep * 2)
    resistantVirusTimeStep = np.zeros(numTimeStep * 2)
    for trial in range(numTrials):
        trialVirus = []
        resistVirus = []
        patient = TreatedPatient(viruses, maxPop)
        # for first set of time step, there is no drug added
        for i in range(1, numTimeStep + 1):
            patient.update()
            trialVirus.append(patient.getTotalPop())
            resistVirus.append((patient.getResistPop(drugToTestList)))
        # for second set of time setp, there is drug added
        patient.addPrescription(drugToTestList[0])
        for i in range(1, numTimeStep + 1):
            patient.update()
            trialVirus.append(patient.getTotalPop())
            resistVirus.append((patient.getResistPop(drugToTestList)))
        virusTimeStep = virusTimeStep + np.array(trialVirus)
        resistantVirusTimeStep = resistantVirusTimeStep + np.array(resistVirus)

    # 4. take the average
    virusTimeStep = np.round(virusTimeStep / numTrials, 1)
    resistantVirusTimeStep = np.round(resistantVirusTimeStep / numTrials, 1)

    answer1 = list(virusTimeStep)
    totalPop = [round(i, 1) for i in answer1]
    answer2 = list(resistantVirusTimeStep)
    resistPop = [round(i, 1) for i in answer2]
    # do this to avoid the stupid grader error
    answer1 = virusTimeStep.tolist()
    totalPop = [round(i, 1) for i in answer1]
    answer2 = resistantVirusTimeStep.tolist()
    resistPop = [round(i, 1) for i in answer2]

    # 5. plot the number of virus vs time step
    pylab.plot(totalPop, label="Total virus population")
    pylab.plot(resistPop, label="Drug resistant virus population")
    pylab.title("Treated Patient simulation")
    pylab.xlabel("Time Steps")
    pylab.ylabel("Average Virus Population")
    pylab.legend(loc="best")
    pylab.show()
