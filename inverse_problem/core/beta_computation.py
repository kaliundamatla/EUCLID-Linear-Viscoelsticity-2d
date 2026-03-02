import numpy as np

def inverse_problem_input_realData_noEps33(nameDir, time, dt, coord, U, conne, Nel, NMeG, NMeK, tauG, tauK):
    NN = coord.shape[0]
    ntsteps = len(time)
    m = np.array([[1], [1], [0]])
    PrD = np.eye(3) - 0.5 * (m @ m.T)
    Dmu = np.array([[2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 1]])

    # Initialize storage
    epsG_test = np.zeros((3, Nel, ntsteps))
    tht_test = np.zeros((1, Nel, ntsteps))
    epsvG = np.zeros((3, NMeG, Nel, ntsteps))
    thtv = np.zeros((1, NMeK, Nel, ntsteps))

    # Output: lists of arrays, one per time step
    betGnm1_out = [np.zeros((3, NMeG, Nel)) for _ in range(ntsteps)]
    betKnm1_out = [np.zeros((1, NMeK, Nel)) for _ in range(ntsteps)]

    for nt in range(ntsteps):
        deformation_x = U[:NN, nt]
        deformation_y = U[NN:, nt]
        Z = np.zeros(2 * NN)
        Z[0::2] = deformation_x
        Z[1::2] = deformation_y

        for ie in range(Nel):
            vrtx = conne[ie, :3]
            X = coord[vrtx, 1]
            Y = coord[vrtx, 2]

            N_xi = np.array([-1, 1, 0])
            N_eta = np.array([-1, 0, 1])
            Jmat = np.array([
                [N_xi @ X, N_xi @ Y],
                [N_eta @ X, N_eta @ Y]
            ])
            Jinv = np.linalg.inv(Jmat)
            grads = np.array([Jinv @ np.array([N_xi[i], N_eta[i]]) for i in range(3)])

            B = np.zeros((3, 6))
            for i in range(3):
                B[0, 2*i]     = grads[i, 0]
                B[1, 2*i + 1] = grads[i, 1]
                B[2, 2*i]     = grads[i, 1]
                B[2, 2*i + 1] = grads[i, 0]

            BD = PrD @ B
            b = (m.T @ B).flatten()
            dofglob = np.ravel(np.vstack([2 * vrtx, 2 * vrtx + 1]), order='F')
            vloce = Z[dofglob]

            epsG = BD @ vloce
            tht = b @ vloce
            epsG_test[:, ie, nt] = epsG
            tht_test[0, ie, nt] = tht

            if nt == 0:
                betGnm1 = np.tile(dt[nt] / (2 * tauG + dt[nt]), (3, 1)) * epsG[:, None]
                betKnm1 = (dt[nt] / (2 * tauK + dt[nt])) * tht
            else:
                epsGnm1 = epsG_test[:, ie, nt - 1]
                epsvGnm1 = epsvG[:, :, ie, nt - 1]
                betGnm1 = (
                    np.tile(dt[nt] / (2 * tauG + dt[nt]), (3, 1)) * (epsGnm1[:, None] - epsvGnm1)
                    + np.tile((2 * tauG) / (2 * tauG + dt[nt]), (3, 1)) * epsvGnm1
                )

                thtnm1 = tht_test[0, ie, nt - 1]
                thtvnm1 = thtv[0, :, ie, nt - 1]
                betKnm1 = (
                    (dt[nt] / (2 * tauK + dt[nt])) * (thtnm1 - thtvnm1)
                    + (2 * tauK / (2 * tauK + dt[nt])) * thtvnm1
                )

            epsvG[:, :, ie, nt] = np.tile(dt[nt] / (2 * tauG + dt[nt]), (3, 1)) * epsG[:, None] + betGnm1
            thtv[0, :, ie, nt] = (dt[nt] / (2 * tauK + dt[nt])) * tht + betKnm1

            betGnm1_out[nt][:, :, ie] = betGnm1
            betKnm1_out[nt][0, :, ie] = betKnm1

    return betGnm1_out, betKnm1_out
