from PyDPOCL import *

if __name__ == '__main__':
    domain_file = 'Ground_Compiler_Library//domains/travel_domain.pddl'

    problem_file_1 = 'Ground_Compiler_Library//domains/travel-to-la.pddl'
    problem_file_2 = 'Ground_Compiler_Library//domains/travel-2.pddl'
    problem_file_3 = 'Ground_Compiler_Library//domains/travel-3.pddl'
    problem_file_4 = 'Ground_Compiler_Library//domains/travel-3.pddl'
    problem_file_5 = 'Ground_Compiler_Library//domains/travel-3.pddl'
    problem_file_6 = 'Ground_Compiler_Library//domains/travel-3.pddl'
    problem_file_7 = 'Ground_Compiler_Library//domains/travel-3.pddl'
    problem_file_8 = 'Ground_Compiler_Library//domains/travel-3.pddl'

    problems = [problem_file_1, problem_file_2, problem_file_3, problem_file_4, problem_file_5, problem_file_6, problem_file_7, problem_file_8]
    d_name = domain_file.split('/')[-1].split('.')[0] # Getting 'travel_domain'
	# for each problem, solve in 1 of 4 ways... but need way to run in different ways

    for prob in problems:
        p_name = prob.split('/')[-1].split('.')[0] #Getting 'problem_name' without extensions
        uploadable_ground_step_library_name = 'Ground_Compiler_Library//' + d_name + '.' + p_name # Meshing names

        RELOAD = 1
        if RELOAD:
            print('Reloading')
            ground_steps = just_compile(domain_file, prob, uploadable_ground_step_library_name)
